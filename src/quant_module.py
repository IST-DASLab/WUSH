"""
QuantLinear -- the quantized inference nn.Module: its weight is stored as packed fp4 (e2m1) codes + per-group
fp8 scales in the transformed space. The forward does a REAL fp4 x fp4 matmul through three interchangeable
backends -- torch (dequant -> @, the reference / fake-quant), triton (fp4_matmul_auto), cuda (qutlass) -- for
both the activation quantization and the matmul, routed by forward(). The quant backend and matmul backend are
chosen independently (mixed), default to the fastest available, and the qutlass / triton imports are optional
(graceful fallback). Instances are produced by the quantization pipeline (quantize_model.quantize_layers).
"""
import warnings

import torch

from block_transform import apply_block_transform
from quantize_fp4 import dequant_xxfp4, rtn_xxfp4, unpack_e2m1

# The Triton kernels are an optional fast path; if the import fails (no Triton build, no CUDA, ...) QuantLinear
# falls back to the pure-PyTorch (torch) backend.
try:
    from triton_kernels.fp4mm import fp4_matmul_auto as _triton_fp4_matmul_auto, to_blocked as _triton_to_blocked, pad_k_to_tile as _triton_pad_k_to_tile
    from triton_kernels.fused import transform_quantize_pack as _triton_transform_quantize_pack
    from triton_kernels.xxfp4 import quantize_pack as _triton_quantize_pack
    _HAS_TRITON_KERNELS: bool = True
except Exception:  # noqa: BLE001 -- any import failure (missing Triton / CUDA) just disables the triton backend
    _HAS_TRITON_KERNELS: bool = False

# The qutlass CUDA kernels are an optional fast path (compiled extension); if the import fails QuantLinear falls
# back to the triton / torch backend. qutlass.ops is qutlass's torch.compile-traceable interface (custom_op +
# register_fake), so the cuda backend traces cleanly under torch.compile.
try:
    from qutlass.ops import fusedQuantizeMx, fusedQuantizeNv, fusedQuantizeWushMx, matmul_mxf4_bf16_tn, matmul_nvf4_bf16_tn
    _HAS_QUTLASS: bool = True
except Exception:  # noqa: BLE001 -- missing compiled qutlass._CUDA just disables the cuda backend
    _HAS_QUTLASS: bool = False


def _scalar_tensor(value: torch.Tensor | float, device: torch.device | None = None) -> torch.Tensor:
    # A 0-dim scalar tensor (optionally on `device`): a Python number -> float32; an existing tensor is kept as-is,
    # preserving its float dtype (e.g. passed explicitly or restored from a checkpoint).
    t: torch.Tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32)
    return t.to(device=device)


class QuantLinear(torch.nn.Module):
    """
    Quantized linear whose weight is stored in NATIVE quantized form (packed fp4 e2m1 codes + per-group fp8
    scales, in the transformed space). forward(x) = alpha * (dequant(quant(x)) @ dequant(weight).T) via a real
    fp4 matmul, routed to one of three backends:
      torch  -- dequantize both operands, then a full-precision @ (the fake-quant reference; always available).
      triton -- quantize the activation + fp4_matmul_auto (pointer / TMA) from triton_kernels.fp4mm.
      cuda   -- qutlass fusedQuantize* + matmul_{mxf4,nvf4}_bf16_tn.
    The quant backend and matmul backend are selected independently (`quant_backend` / `matmul_backend`, each
    None => the fastest available given the data / format / scales). forward resolves the quant and matmul
    backends independently and runs them via _run (same backend or mixed).

    The constructor is a SUPERSET of torch.nn.Linear's (in_features, out_features, bias, device, dtype), so
    QuantLinear drops in at construction; the quantized weight arrives via the keyword-only tensors; otherwise
    (in_features, out_features, group_size) build shape-compatible placeholder buffers. `bias` is accepted only
    for nn.Linear-constructor compatibility (the quantized forward never applies a bias; self.bias is None).

    Scales are per-operand: `weight_global_scale` describes the stored WEIGHT; `activation_global_scale` /
    `activation_scale_scale` describe how the ACTIVATION is quantized at forward (gs None => inherit the weight's;
    ss None => 1.). Only the matmul uses gs (alpha = weight_gs * activation_gs); ss only affects
    quantization. The stored weight is already quantized, so the weight's ss never enters the matmul.

    State (round-trips through the state_dict, moves with .to(device); the quantized weight AND the scalar
    scales are UNCHANGED by .to(dtype) -- kept at their stored dtype by the _apply override). Buffers:
      e2m1        : (out, in // 2)  uint8 -- packed float4_e2m1fn_x2 codes (uint8 view: fp4 has incomplete eager
                                             CUDA ops, so uint8 survives copy / serialization / .to(dtype);
                                             _weight re-views it as fp4 for the kernels).
      scale_quant : (out, in // G)  native fp8 (e8m0 => mxfp4, e4m3 => nvfp4) -- per-group scale; its dtype IS
                                             the scale's fp8 format (read directly, no separate metadata).
      transform   : (in // G, G, G) per-block OR (G, G) shared/repeated (C=1) OR None -- input-dim transform.
      bias        : always None.
      weight_global_scale / activation_global_scale / activation_scale_scale : 0-dim float32 scalar buffers --
                                             global scales + activation scale-scale; serialize with the state_dict,
                                             move with .to(device), and stay put under .to(dtype) (calibration
                                             constants, like the quantized weight).
    group_size / in_features / out_features are derived from the buffer shapes.
    """

    def __init__(
            self,
            in_features: int = 0,
            out_features: int = 0,
            bias: bool = True,  # bias is accepted for nn.Linear compat but never materialized
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
            *,
            transform: torch.Tensor | None = None,
            e2m1: torch.Tensor | None = None,  # (out, in // 2) packed weight codes (float4_e2m1fn_x2 or its uint8 view)
            scale_quant: torch.Tensor | None = None,  # (out, in // G) per-group weight scale (fp8 or its uint8 view)
            scale_dtype: torch.dtype | None = None,  # the scale's fp8 dtype (e8m0 => mxfp4, e4m3 => nvfp4); if None, taken from scale_quant.dtype (e8m0 for an empty skeleton)
            group_size: int = 32,  # quant group G; sizes scale_quant when no tensor is passed (else derived from the tensors)
            weight_global_scale: torch.Tensor | float = 1.,  # weight gs -- folded into the matmul alpha (weight_gs * activation_gs)
            activation_global_scale: torch.Tensor | float | None = None,  # activation gs (None => inherit the weight's)
            activation_scale_scale: torch.Tensor | float | None = None,  # activation ss (None => 1.)
            quant_backend: str | None = None,  # 'torch' / 'triton' / 'cuda'; None => fastest available
            matmul_backend: str | None = None,  # 'torch' / 'triton' / 'cuda'; None => fastest available
    ) -> None:
        super().__init__()
        self.quant_backend: str | None = quant_backend
        self.matmul_backend: str | None = matmul_backend
        sdt: torch.dtype = scale_dtype if scale_dtype is not None else (scale_quant.dtype if scale_quant is not None else torch.float8_e8m0fnu)
        device = e2m1.device if e2m1 is not None else device  # keep placeholders on the data's device when given
        # e2m1 is a uint8 view: fp4 (float4_e2m1fn_x2) has incomplete eager CUDA ops (no zeros/copy_/cast), so uint8
        # lets us .contiguous()/serialize/move it and keeps .to(dtype) off it (_weight re-views it as fp4 for the
        # kernels). scale_quant is stored in its NATIVE fp8 (e8m0/e4m3) -- fp8 ops are complete; _apply keeps both
        # immune to .to(dtype).
        if e2m1 is None:  # no data -> shape-compatible placeholders, filled by load_state_dict
            self.register_buffer('e2m1', torch.zeros(out_features, in_features // 2, dtype=torch.uint8, device=device))
            self.register_buffer('scale_quant', torch.zeros(out_features, in_features // group_size, dtype=sdt, device=device))
        else:
            out_features, in_features = e2m1.size(-2), e2m1.size(-1) * 2  # 2 fp4 codes per packed byte
            group_size = in_features // scale_quant.size(-1)  # derived from the tensors when data is given
            self.register_buffer('e2m1', e2m1.view(dtype=torch.uint8).reshape(out_features, in_features // 2).contiguous())
            self.register_buffer('scale_quant', scale_quant.view(dtype=sdt).reshape(out_features, in_features // group_size).contiguous())
        # transform (a general per-block linear transform, not necessarily orthogonal): None = no transform;
        # 2D (G, G) = shared/repeated (C=1); 3D (in // G, G, G) = per-block.
        self.register_buffer('transform', transform.contiguous() if transform is not None else None)
        # bias: accepted for torch.nn.Linear-constructor compatibility but NEVER used in the forward, so no tensor.
        self.register_buffer('bias', None)
        # global scales / scale-scale as 0-dim buffers (serialize with the state_dict, move with .to(device)). The
        # weight gs folds into the matmul alpha (weight_gs * activation_gs); the activation gs inherits the weight's
        # when None. ss only affects quantization -- it never enters the matmul (the weight is already quantized), so
        # the weight's ss is not stored and the activation ss is taken directly (None => 1.).
        weight_gs: torch.Tensor = _scalar_tensor(weight_global_scale, device=device)
        self.register_buffer('weight_global_scale', weight_gs)
        self.register_buffer('activation_global_scale', weight_gs.clone() if activation_global_scale is None else _scalar_tensor(activation_global_scale, device=device))
        self.register_buffer('activation_scale_scale', _scalar_tensor(1. if activation_scale_scale is None else activation_scale_scale, device=device))

    # shapes / group size are derived from the buffers, so they are correct again automatically after a load.
    @property
    def out_features(self) -> int:
        return self.e2m1.size(0)

    @property
    def in_features(self) -> int:
        return self.e2m1.size(-1) * 2

    @property
    def group_size(self) -> int:
        return self.in_features // self.scale_quant.size(-1)

    @property
    def _weight(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.e2m1.view(dtype=torch.float4_e2m1fn_x2), self.scale_quant

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None:
        # placeholders may differ from the checkpoint: 0-shaped (e2m1/scale_quant) or None (transform when the
        # skeleton has no transform). For each buffer the checkpoint provides, (re)create a local tensor matching
        # its shape/dtype -- on the module's device -- so the default in-place copy succeeds.
        ref_device: torch.device = self.e2m1.device
        for name in list(self._buffers.keys()):
            key: str = prefix + name
            if key in state_dict:
                buffer: torch.Tensor | None = self._buffers[name]
                source: torch.Tensor = state_dict[key]
                if buffer is None or buffer.shape != source.shape or buffer.dtype != source.dtype:
                    self._buffers[name] = torch.empty_like(source, device=buffer.device if buffer is not None else ref_device)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def _apply(self, fn, recurse=True):
        # The stored quantized weight (e2m1 / scale_quant, uint8) and the scalar scales (calibration constants)
        # keep their stored dtype+VALUE under .to(dtype), while still moving with .to(device) and serializing.
        # transform is NOT here -- it follows the activation dtype. Save the originals, run the generic apply, then
        # wherever it changed a dtype, restore the original tensor re-placed on the new device (a re-cast of the
        # already-cast buffer would NOT recover the lost precision). A pure device move leaves the dtype intact, so
        # the restore is skipped and the large e2m1 is not copied twice.
        immune: tuple = ('e2m1', 'scale_quant', 'weight_global_scale', 'activation_global_scale', 'activation_scale_scale')
        saved: dict = {n: self._buffers[n] for n in immune if self._buffers.get(n) is not None}
        super()._apply(fn, recurse=recurse)
        for n, orig in saved.items():
            cur: torch.Tensor = self._buffers[n]
            if cur.dtype != orig.dtype:  # a dtype cast happened -> keep the original value/dtype, at the new device
                self._buffers[n] = orig.to(device=cur.device)
        return self

    # ----------------------------------------------------------------------- #
    # Activation quantization backends: (x2d) -> (packed e2m1, row-major scale), using the activation gs / ss.
    # The stored weight is already in this (packed, scale) form, so only the activation is quantized per-forward.
    # ----------------------------------------------------------------------- #
    def _quantize_torch(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        transform: torch.Tensor | None = None if self.transform is None else self.transform.to(dtype=x.dtype)
        if transform is not None and transform.dim() == 2:  # shared (C=1): expand to a per-block stack for the einsum
            transform = transform.unsqueeze(0).expand(self.in_features // self.group_size, -1, -1)
        x_t: torch.Tensor = x if transform is None else apply_block_transform(x=x, transform=transform, is_inverse_transpose=False, high_dtype=torch.float32, round_dtype=x.dtype)
        r: dict = rtn_xxfp4(x=x_t, group_size=self.group_size, scale_dtype=self.scale_quant.dtype, scale_scale=self.activation_scale_scale, global_scale=self.activation_global_scale, fp4_rounding_mode='even', high_dtype=torch.float32, round_dtype=x.dtype)
        return r['e2m1'], r['scale_quant']

    def _quantize_triton(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        transform: torch.Tensor | None = None if self.transform is None else self.transform.to(dtype=x.dtype)
        if transform is None:  # transform_quantize_pack accepts a 2D (repeated) or 3D (per-block) transform directly
            return _triton_quantize_pack(x=x, group_size=self.group_size, scale_dtype=self.scale_quant.dtype, scale_scale=self.activation_scale_scale, global_scale=self.activation_global_scale)
        return _triton_transform_quantize_pack(x=x, transform=transform, scale_dtype=self.scale_quant.dtype, scale_scale=self.activation_scale_scale, global_scale=self.activation_global_scale)

    def _quantize_cuda(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # qutlass fuses the transform (operand B of a CUTLASS GEMM) + quantization. Requires bf16 input. The
        # compiled kernels use the OLD (standard NVFP4) convention -- gs is the divide-in-decode scale, i.e. the
        # reciprocal of our new multiply-in-decode gs: mxfp4 bakes gs=3 (== 1/(1/3)); nvfp4 takes gs at runtime,
        # so we pass 1/activation_global_scale (== old gs). Both bake ss (4 / 6) -- the resolver routes here only when valid.
        device: torch.device = x.device
        x_bf16: torch.Tensor = x.to(dtype=torch.bfloat16)
        g: int = self.group_size
        c: int = self.in_features // g
        # qutlass quantizes x @ b (operand B of a CUTLASS GEMM); our convention is x @ transform.T, so B = transform.T.
        if self.scale_quant.dtype == torch.float8_e8m0fnu:  # mxfp4
            if self.transform is not None and self.transform.dim() == 3:  # per-block -> WushMx (B = (G, C*G), = w[c].T per block)
                packed, scale = fusedQuantizeWushMx(x_bf16, self.transform.to(dtype=torch.bfloat16).reshape(-1, g).transpose(-2, -1).contiguous())
            else:  # shared 2D or no transform (identity) -> Mx
                b: torch.Tensor = torch.eye(g, dtype=torch.bfloat16, device=device) if self.transform is None else self.transform.to(dtype=torch.bfloat16).transpose(-2, -1).contiguous()
                packed, scale = fusedQuantizeMx(x_bf16, b, method='abs_max')
        else:  # nvfp4 -- shared 2D transform only (no per-block nv kernel)
            packed, scale = fusedQuantizeNv(
                x_bf16,
                self.transform.to(dtype=torch.bfloat16).transpose(-2, -1).contiguous(),
                (1. / self.activation_global_scale).reshape(1).to(dtype=torch.float32, device=device),
            )  # old-convention kernel: feed 1/gs (= old gs); qutlass wants a (1,)-shaped global_scale (bindings: dim==1 && size(0)==1)
        # qutlass's fused-quant kernel writes the scale ROW-MAJOR but over-allocates the buffer to the 128x4
        # SF-tile multiples (rows -> ceil(M/128)*128, group-cols -> ceil(c/4)*4; see
        # qutlass.utils.get_padded_shape_mx/_nv). Slice that padding off so every backend shares ONE unpadded,
        # logical (M, in // G) row-major scale -- torch/triton consume it directly; the cuda matmul zero-pads K to
        # a multiple of 128 (pad_k_to_tile) and re-swizzles via to_blocked.
        return packed, scale[:x.size(0), :c].contiguous()

    # ----------------------------------------------------------------------- #
    # Matmul backends: (a, a_scale, out_dtype) -> alpha * (dequant(a) @ dequant(weight).T). The weight operand
    # (self._weight) and the scales (alpha = activation_gs * weight_gs) are read from self; a / a_scale are the
    # packed e2m1 activation + its row-major native-fp8 group scale.
    # ----------------------------------------------------------------------- #
    def _matmul_torch(self, a: torch.Tensor, a_scale: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        # fold each gs into its dequant -> bit-identical to the legacy fake-quant forward (the reference).
        b, b_scale = self._weight
        a_fq: torch.Tensor = dequant_xxfp4(e2m1=a, scale_quant=a_scale, global_scale=self.activation_global_scale, dtype=out_dtype, high_dtype=torch.float32, round_dtype=out_dtype)
        b_fq: torch.Tensor = dequant_xxfp4(e2m1=b, scale_quant=b_scale, global_scale=self.weight_global_scale, dtype=out_dtype, high_dtype=torch.float32, round_dtype=out_dtype)
        return a_fq @ b_fq.transpose(-2, -1)

    def _matmul_triton(self, a: torch.Tensor, a_scale: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        b, b_scale = self._weight
        return _triton_fp4_matmul_auto(a, b, a_scale, b_scale, self.activation_global_scale * self.weight_global_scale, out_dtype=out_dtype)

    def _matmul_cuda(self, a: torch.Tensor, a_scale: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        b, b_scale = self._weight
        a, b, a_scale, b_scale = _triton_pad_k_to_tile(a, b, a_scale, b_scale, self.group_size)  # K -> multiple of 128 (qutlass needs whole 128x4 SF tiles); inert (fp4 0x00 = +0.0). UNPACK ORDER must match the return
        a_u8, b_u8 = a.view(dtype=torch.uint8), b.view(dtype=torch.uint8)  # qutlass takes uint8-packed operands
        a_blk, b_blk = _triton_to_blocked(a_scale), _triton_to_blocked(b_scale)  # row-major -> swizzled (flat, native fp8 dtype)
        alpha: torch.Tensor = (self.activation_global_scale * self.weight_global_scale).to(dtype=torch.float32, device=a.device)  # 0-d (qutlass reads alpha.data_ptr(), shape-agnostic)
        if self.scale_quant.dtype == torch.float8_e8m0fnu:  # mxfp4 -- the kernel takes the flat blocked scale
            out: torch.Tensor = matmul_mxf4_bf16_tn(a_u8, b_u8, a_blk, b_blk, alpha)
        else:  # nvfp4 -- reshape the blocked scale to (-1, K // 16); K is the PADDED K (the padded operand's last dim * 2)
            k: int = a_u8.size(-1) * 2
            out = matmul_nvf4_bf16_tn(a_u8, b_u8, a_blk.view(-1, k // 16), b_blk.view(-1, k // 16), alpha)
        return out.to(dtype=out_dtype)  # qutlass returns bf16

    # ----------------------------------------------------------------------- #
    # The router: resolve the quant / matmul backends, then run them (same backend or mixed).
    # ----------------------------------------------------------------------- #
    def _run(self, quant_backend: str, matmul_backend: str, x: torch.Tensor) -> torch.Tensor:
        x2d: torch.Tensor = x.reshape(-1, self.in_features)  # (..., in) -> (M, in)
        a_packed, a_scale = getattr(self, f'_quantize_{quant_backend}')(x2d)
        y: torch.Tensor = getattr(self, f'_matmul_{matmul_backend}')(a_packed, a_scale, x.dtype)  # weight operand + scales read from self
        return y.reshape(*x.shape[:-1], self.out_features)  # (M, out) -> (..., out); bias is intentionally NOT applied

    def _available(self, role: str, backend: str, dtype: torch.dtype, device: torch.device) -> bool:
        # role in {'quant', 'matmul'}; can this backend run this role on (dtype, device) given availability + constraints?
        if backend == 'torch':
            return True
        if device.type != 'cuda':
            return False
        if backend == 'triton':
            return _HAS_TRITON_KERNELS
        # backend == 'cuda' (qutlass): needs the compiled extension + bf16 (qutlass quant/matmul are bf16-only).
        if not (_HAS_QUTLASS and dtype == torch.bfloat16):
            return False
        if role == 'matmul':  # uses to_blocked + pad_k_to_tile (from fp4mm); K is zero-padded to a multiple of 128, so any in_features works
            return _HAS_TRITON_KERNELS
        if self.scale_quant.dtype == torch.float8_e8m0fnu:  # mxfp4 quant: gs/ss baked into the kernel
            return (self.activation_global_scale.isclose(_scalar_tensor(1. / 3., device=device), rtol=1e-6).item()
                    and self.activation_scale_scale.isclose(_scalar_tensor(4., device=device), rtol=1e-6).item())
        # nvfp4 quant: ss=6 baked in, gs runtime; needs a shared 2D transform (no per-block nv kernel, no identity)
        return (self.activation_scale_scale.isclose(_scalar_tensor(6., device=device), rtol=1e-6).item()
                and self.transform is not None and self.transform.dim() == 2)

    def _resolve(self, role: str, x: torch.Tensor) -> str:
        explicit: str | None = self.quant_backend if role == 'quant' else self.matmul_backend
        if explicit is not None:
            if self._available(role=role, backend=explicit, dtype=x.dtype, device=x.device):
                return explicit
            warnings.warn(f'{role}_backend={explicit!r} unavailable for this input; falling back', stacklevel=3)
        for backend in 'cuda', 'triton', 'torch':  # fastest first
            if self._available(role=role, backend=backend, dtype=x.dtype, device=x.device):
                return backend
        return 'torch'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._run(self._resolve('quant', x), self._resolve('matmul', x), x)


# =========================================================================== #
# Unit test  (run from the source root:  CUDA_VISIBLE_DEVICES=<gpu> python -m quant_module)
# =========================================================================== #
def _unit_test(device: torch.device = torch.device('cuda')) -> None:
    """
    QuantLinear: the torch / triton / cuda quant + matmul backends must agree, in any mix. Each quant backend
    produces the same (packed, scale) where the kernels are bit-exact (no-transform: all three; cuda-quant
    matches the triton tensor-core quant; nvfp4 cuda-quant is bit-exact-modulo-rcp.approx); each matmul backend
    reproduces alpha * (dequant(a) @ dequant(b).T) against an fp64 reference built from the SAME operands. Also:
    a state_dict round-trips into a fresh EMPTY skeleton (restoring scale_dtype + the per-operand scales), and
    the quantized weight survives .to(device) and -- unchanged -- .to(dtype).
    """
    torch.manual_seed(seed=0)
    e8m0, e4m3 = torch.float8_e8m0fnu, torch.float8_e4m3fn
    out_features, in_features = 512, 2048

    def make_transform(kind: str, g: int) -> torch.Tensor | None:
        if kind == 'none':
            return None
        if kind == 'shared2d':
            return torch.randn(g, g, dtype=torch.bfloat16, device=device) * g ** -.5
        return torch.randn(in_features // g, g, g, dtype=torch.bfloat16, device=device) * g ** -.5  # per-block

    failures: list = []

    def check(ok: bool, tag: str, detail: str = '') -> None:
        print(f"{'ok  ' if ok else 'FAIL'}  {tag}{('  | ' + detail) if detail else ''}")
        if not ok:
            failures.append(tag)

    def codes(packed: torch.Tensor) -> torch.Tensor:
        return unpack_e2m1(packed.view(dtype=torch.float4_e2m1fn_x2), dtype=torch.float64)

    def agree(p0: torch.Tensor, s0: torch.Tensor, p1: torch.Tensor, s1: torch.Tensor) -> tuple[float, float]:
        code_eq: float = codes(p0).eq(codes(p1)).to(dtype=torch.float64).mean().item()
        scale_eq: float = s0.view(dtype=torch.uint8).eq(s1.view(dtype=torch.uint8)).to(dtype=torch.float64).mean().item()
        return code_eq, scale_eq

    def rel_l2(y: torch.Tensor, ref: torch.Tensor) -> float:
        return (torch.linalg.vector_norm(y.to(dtype=torch.float64) - ref) / torch.linalg.vector_norm(ref)).item()

    # (label, scale_dtype, group_size, weight gs, weight ss, activation gs, activation ss, transform kind)
    configs: list = [
        ('mxfp4/perblock',    e8m0, 32, 1./3., 4., 1./3., 4., 'perblock'),
        ('mxfp4/shared2d',    e8m0, 32, 1./3., 4., 1./3., 4., 'shared2d'),
        ('mxfp4/notransform', e8m0, 32, 1./3., 4., 1./3., 4., 'none'),
        ('nvfp4/perblock',    e4m3, 16, .1, 6., .1, 6., 'perblock'),  # cuda-quant off (per-block) -> mixed paths
        ('nvfp4/shared2d',    e4m3, 16, 1./6., 6., 1./6., 6., 'shared2d'),  # cuda-quant on
        ('nvfp4/mixed_gs',    e4m3, 16, .1, 6., 1./8., 6., 'shared2d'),  # gs_a != gs_w (per-operand alpha)
    ]
    for label, scale_dtype, g, gs_w, ss_w, gs_a, ss_a, kind in configs:
        transform: torch.Tensor | None = make_transform(kind, g)
        weight: torch.Tensor = torch.randn(out_features, in_features, dtype=torch.bfloat16, device=device)
        res: dict = rtn_xxfp4(x=weight, group_size=g, scale_dtype=scale_dtype, scale_scale=ss_w, global_scale=gs_w, fp4_rounding_mode='even', high_dtype=torch.float32, round_dtype=torch.bfloat16)
        ql: QuantLinear = QuantLinear(transform=transform, e2m1=res['e2m1'], scale_quant=res['scale_quant'], scale_dtype=scale_dtype, weight_global_scale=gs_w, activation_global_scale=gs_a, activation_scale_scale=ss_a).to(device=device)
        assert ql.group_size == g and _scalar_tensor(gs_w).item() == ql.weight_global_scale.item() and _scalar_tensor(gs_a).item() == ql.activation_global_scale.item() and _scalar_tensor(ss_a).item() == ql.activation_scale_scale.item(), (label, 'metadata')

        for n_lead in (4, 128), (3, 17):  # full + partial-M (M=512 / 51; exercises to_blocked padding)
            x: torch.Tensor = torch.randn(*n_lead, in_features, dtype=torch.bfloat16, device=device)
            x2d: torch.Tensor = x.reshape(-1, in_features)
            quant_avail: list = [b for b in ('torch', 'triton', 'cuda') if ql._available(role='quant', backend=b, dtype=x.dtype, device=x.device)]
            matmul_avail: list = [b for b in ('torch', 'triton', 'cuda') if ql._available(role='matmul', backend=b, dtype=x.dtype, device=x.device)]
            quants: dict = {b: getattr(ql, f'_quantize_{b}')(x2d) for b in quant_avail}

            # L1 -- quant backends agree. torch (einsum transform) differs from the tensor-core kernels only by
            # transform reduction order, so torch==triton is asserted bit-exact only with NO transform; the
            # triton<->cuda (both tensor-core) agreement is the cuda-quant correctness check.
            tag: str = f'{label}[M={x2d.size(0)}]'
            if kind == 'none' and 'triton' in quants:
                ce, se = agree(*quants['torch'], *quants['triton'])
                check(ce == 1. and se == 1., f'L1 quant torch==triton {tag}', f'code {ce:.4f} scale {se:.4f}')
            if 'cuda' in quants and 'triton' in quants:
                ce, se = agree(*quants['triton'], *quants['cuda'])
                bar: float = 1. if scale_dtype is e8m0 else .999  # nvfp4 cuda uses rcp.approx -> ~bit-exact
                check(ce >= bar and se >= bar, f'L1 quant triton==cuda {tag}', f'code {ce:.4f} scale {se:.4f}')

            # L2 -- each matmul backend reproduces alpha * (dequant(a) @ dequant(b).T) from the SAME operands.
            b_packed, b_scale = ql._weight
            w_ref: torch.Tensor = dequant_xxfp4(e2m1=b_packed, scale_quant=b_scale, global_scale=gs_w, dtype=torch.float64)
            for qb in quant_avail:
                a_packed, a_scale = quants[qb]
                a_ref: torch.Tensor = dequant_xxfp4(e2m1=a_packed, scale_quant=a_scale, global_scale=gs_a, dtype=torch.float64)
                ref: torch.Tensor = a_ref @ w_ref.transpose(-2, -1)
                for mb in matmul_avail:
                    y: torch.Tensor = getattr(ql, f'_matmul_{mb}')(a_packed, a_scale, torch.bfloat16)
                    check(y.dtype == torch.bfloat16 and rel_l2(y, ref) < 2e-2, f'L2 {qb}->{mb} {tag}', f'rel_l2 {rel_l2(y, ref):.2e}')

            # the full forward (routed, fastest available, leading dims preserved) must match the reference too
            ya: torch.Tensor = ql(x).reshape(-1, out_features)
            a_ref0: torch.Tensor = dequant_xxfp4(e2m1=quants['torch'][0], scale_quant=quants['torch'][1], global_scale=gs_a, dtype=torch.float64)
            fref: torch.Tensor = a_ref0 @ w_ref.transpose(-2, -1)
            check(rel_l2(ya, fref) < 5e-2, f'forward {tag}', f'rel_l2 {rel_l2(ya, fref):.2e}')

        # non-bf16 input must route to triton (not cuda), and run
        if _HAS_TRITON_KERNELS:
            xf: torch.Tensor = torch.randn(8, in_features, dtype=torch.float16, device=device)
            check(ql._resolve('matmul', xf) != 'cuda' and ql._resolve('quant', xf) != 'cuda', f'{label} non-bf16 -> not cuda')
            _ = ql(xf)

    # Odd in_features (a multiple of the group but NOT of group*4): the matmul fast paths must still run -- K is
    # zero-padded to a multiple of 128 (pad_k_to_tile). On a box without qutlass this exercises the triton matmul
    # e2e; with qutlass it also lights up the cuda backend (the dropped (in//G) % 4 == 0 gate in _available).
    for olabel, odt, og, ogs, oss in ('mxfp4/oddK', e8m0, 32, 1. / 3., 4.), ('nvfp4/oddK', e4m3, 16, .1, 6.):
        in_odd: int = 2080  # % 32 == 0 and % 16 == 0, but % 128 != 0 and % 64 != 0 -> the odd-K case (in // g not /4)
        ow: torch.Tensor = torch.randn(out_features, in_odd, dtype=torch.bfloat16, device=device)
        ores: dict = rtn_xxfp4(x=ow, group_size=og, scale_dtype=odt, scale_scale=oss, global_scale=ogs, fp4_rounding_mode='even', high_dtype=torch.float32, round_dtype=torch.bfloat16)
        qo: QuantLinear = QuantLinear(transform=torch.randn(og, og, dtype=torch.bfloat16, device=device) * og ** -.5, e2m1=ores['e2m1'], scale_quant=ores['scale_quant'], scale_dtype=odt, weight_global_scale=ogs, activation_global_scale=ogs, activation_scale_scale=oss).to(device=device)
        check((qo.in_features // og) % 4 != 0, f'{olabel} in_features not (group*4)-aligned', f'in={in_odd}, in//g={in_odd // og}')
        check((not _HAS_QUTLASS) or qo._available('matmul', 'cuda', torch.bfloat16, device), f'{olabel} cuda matmul available for odd in_features (qutlass)')  # the dropped gate -- only observable with qutlass
        xo: torch.Tensor = torch.randn(128, in_odd, dtype=torch.bfloat16, device=device)  # M=128 -> prefill path
        bp, bsc = qo._weight
        wref: torch.Tensor = dequant_xxfp4(e2m1=bp, scale_quant=bsc, global_scale=ogs, dtype=torch.float64)
        ap, asc = qo._quantize_torch(xo)
        oref: torch.Tensor = dequant_xxfp4(e2m1=ap, scale_quant=asc, global_scale=ogs, dtype=torch.float64) @ wref.transpose(-2, -1)
        for mb in [b for b in ('torch', 'triton', 'cuda') if qo._available('matmul', b, xo.dtype, device)]:
            yo: torch.Tensor = getattr(qo, f'_matmul_{mb}')(ap, asc, torch.bfloat16)
            check(yo.dtype == torch.bfloat16 and rel_l2(yo, oref) < 2e-2, f'{olabel} matmul {mb} [M=128]', f'rel_l2 {rel_l2(yo, oref):.2e}')
        _ = qo(xo)  # routed forward (fastest available backend) must run end-to-end for odd in_features

    # state_dict round-trips into a fresh EMPTY skeleton.
    transform = make_transform('perblock', 32)
    res = rtn_xxfp4(x=torch.randn(out_features, in_features, dtype=torch.bfloat16, device=device), group_size=32, scale_dtype=e8m0, scale_scale=4., global_scale=1. / 3., fp4_rounding_mode='even', high_dtype=torch.float32, round_dtype=torch.bfloat16)
    ql = QuantLinear(transform=transform, e2m1=res['e2m1'], scale_quant=res['scale_quant'], scale_dtype=e8m0, weight_global_scale=1. / 3., activation_global_scale=1. / 2., activation_scale_scale=4.).to(device=device)
    x = torch.randn(4, 128, in_features, dtype=torch.bfloat16, device=device)
    y0: torch.Tensor = ql(x)
    state_dict: dict = ql.state_dict()
    check(state_dict['e2m1'].dtype == torch.uint8 and state_dict['scale_quant'].dtype == e8m0, 'state_dict dtypes (e2m1 uint8, scale fp8)')
    check(state_dict['activation_global_scale'].item() == _scalar_tensor(1. / 2.).item(), 'scale buffers saved')
    check('bias' not in state_dict, 'no bias stored')
    ql_loaded: QuantLinear = QuantLinear().to(device=device)
    ql_loaded.load_state_dict(state_dict)
    check(ql_loaded.scale_quant.dtype == e8m0 and ql_loaded.group_size == 32 and ql_loaded.activation_global_scale.item() == _scalar_tensor(1. / 2.).item() and ql_loaded.activation_scale_scale.item() == _scalar_tensor(4.).item(), 'metadata restored')
    check(ql_loaded(x).equal(y0), 'empty-skeleton load reproduces output')

    # .to(dtype) leaves the quantized weight byte-for-byte unchanged AND the scalar scales fp32 (immune, via
    # _apply); transform DOES follow the dtype. .to(device) moves everything unchanged.
    e2m1_before, scale_before = ql.e2m1.clone(), ql.scale_quant.view(torch.uint8).clone()  # compare scale bytes via uint8 (avoid fp8 .equal)
    ql.to(dtype=torch.float16)
    check(ql.e2m1.dtype == torch.uint8 and ql.scale_quant.dtype == e8m0 and ql.e2m1.equal(e2m1_before) and ql.scale_quant.view(torch.uint8).equal(scale_before), '.to(dtype) keeps quantized buffers')
    check(ql.scale_quant.dtype == e8m0 and ql.transform.dtype == torch.float16, '.to(dtype) metadata + transform cast')
    check(ql.weight_global_scale.dtype == torch.float32 and ql.activation_global_scale.dtype == torch.float32 and ql.activation_scale_scale.dtype == torch.float32, '.to(dtype) keeps scale buffers fp32 (immune)')
    _ = ql(x.to(dtype=torch.float16))
    ql.cpu()
    check(ql.e2m1.device.type == 'cpu' and ql.e2m1.cpu().equal(e2m1_before.cpu()), '.to(device) moves quantized buffers unchanged')
    check(ql.weight_global_scale.device.type == 'cpu' and ql.weight_global_scale.dtype == torch.float32, '.to(device) moves scale buffers (dtype intact)')

    print('-' * 60)
    assert not failures, failures
    print('Unit test passed.')


if __name__ == '__main__':
    _unit_test(device=torch.device('cuda'))
