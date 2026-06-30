"""
Fused block transform + XXFP4 (MXFP4 / NVFP4) group quantization, in a single pass.

Composes shared compute directly -- the load + transform comes from transform.py and the per-group
quant from xxfp4.py; fused inserts the quant between them and writes its own tile-shaped stores:
  - transform.load_transform : map program id -> tile (the bandwidth swizzle), load the x slab + apply
        the block transform (per-block (C, GT, GT) OR a repeated 2D (GT, GT) matrix, exactly as
        transform.block_transform) -> the transformed slab + the tile's (rm, c0) coords;
  - xxfp4.quant_group        : per-group AbsMax scale (e8m0 / hardware float8) + the scaled values, which
        e2m1.round_pack / round_value then RNE-encode to e2m1 bytes / fake-quant grid values;
  - three 2D tl.stores        : fake-quant values / packed codes / per-group scale, each addressing
        this program's swizzled TILE_M x NT tile directly (the store layout is the one thing the
        two quant kernels do NOT share, so it stays local -- no offset / mask tensors are threaded in).
Plus transform's autotune set + shape-aware prune (TILE_CONFIGS / prune_configs) -- so a fused call tunes and
behaves like the transform kernel, then quantizes in registers.

The transform block size GT and the quant group size GQ are independent powers of two: pass quant_group_size
either SMALLER than GT (each transformed block splits into GT//GQ groups, e.g. transform GT=128 quantize 32)
or LARGER than GT (one group spans GQ//GT blocks, e.g. transform GT=16 quantize 32) -- the latter needs a
per-block (C>1) transform (see transform_quantize). GQ defaults to GT (one block = one group); the group size
sets the format (32 for mxfp4, 16 for nvfp4). Fusing skips a DRAM round-trip of the transformed activations;
the +pack variant also writes 4x less (fp4 codes + tiny scales).

Decoupled group sizes (quant group GQ < transform block GT) stay fully fused at the FULL TILE_M config set on
every device. On cc < 9 (sm8x) as_triton_fp8_view hands the e4m3 scale to the kernel as an INT8 view: its
byte is built with pure integer ops (xxfp4.to_e4m3 -> int8) and stored with a plain tl.store -- exactly like
the e8m0 (mxfp4) path. This emits NO tl.float8e4nv, so the kernel compiles on sm86 (where fp8e4nv is unsupported)
and never feeds an fp8 store into the tl.dot convert_layout that a Triton codegen bug miscompiles on sm_89 at
TILE_M >= 32 -- so all TILE_M configs are bit-exact (verified vs the reference). (The simpler original
workaround for that bug was to pin the autotuner to TILE_M == 16; see the BUG/FIX note at _kernel's scale
store for both remedies.) sm_90+ (Hopper / Blackwell,
fp8_native_rne) keeps the e4m3 scale as float8_e4m3fn and uses the native RNE fp8 cvt + fp8 store (the bug is
absent there; verified bit-exact on sm_120 at TILE_M=32). The coupled case (GQ == GT) is unaffected.

Packed output (packed float4_e2m1fn_x2 (...,K//2), scale (...,K//GQ)) is byte-identical to xxfp4.quantize_pack
on the transformed input, so the inverse is xxfp4.dequantize.
"""

import functools

import torch
import triton
import triton.language as tl

from triton_kernels.e2m1 import round_pack, dequant, round_value
from triton_kernels.guard import device_guard, has_capability
from triton_kernels.transform import load_transform, TILE_CONFIGS, prune_configs, block_transform  # load + transform; shared autotune + GT-keyed prune (block_transform also used by the unit test)
from triton_kernels.xxfp4 import quant_group, compat_shift, as_triton_fp8_view  # per-group quant + fp8 scale view (as_triton_fp8_view picks the per-device view)


@triton.autotune(configs=TILE_CONFIGS, key=['C', 'GT', 'GQ', 'ISIZE', 'DO_DEQUANT', 'DO_PACK', 'HW_CVT'], prune_configs_by={'early_config_prune': prune_configs})
@triton.jit
def _kernel(
        x_ptr, t_ptr, y_ptr, q_ptr, s_ptr, gs_ptr, ss_ptr,
        M, C: tl.constexpr, GT: tl.constexpr, GQ: tl.constexpr,  # GT = transform block, GQ = quant group (GQ <= GT, GT % GQ == 0)
        stride_xm, stride_xk, stride_tc, stride_ti, stride_tj,
        stride_ym, stride_yk, stride_qm, stride_qk, stride_sm, stride_sc,
        ISIZE: tl.constexpr,
        DO_DEQUANT: tl.constexpr, DO_PACK: tl.constexpr, HW_CVT: tl.constexpr,
        TILE_M: tl.constexpr, WIDTH: tl.constexpr,
):
    NT: tl.constexpr = WIDTH // GT  # transform blocks per program (drives the load + transform tiling)
    NQ: tl.constexpr = WIDTH // GQ  # quant groups per program -- the slab's WIDTH cols split into NQ groups of GQ
    IS_FP64: tl.constexpr = ISIZE == 8
    ACC: tl.constexpr = tl.float64 if IS_FP64 else tl.float32
    USE_HW: tl.constexpr = HW_CVT and not IS_FP64  # the e2m1 cvt is f32-only; fp64 keeps the libdevice round
    USE_DOT: tl.constexpr = GT >= 16 and not IS_FP64  # tl.dot needs contraction dim (= GT) >= 16; fp64 stays manual
    gs = tl.load(gs_ptr).to(ACC)  # gs_ptr/ss_ptr -> 0-d fp64 tensors, full precision
    ss = tl.load(ss_ptr).to(ACC)

    # load + block transform (transform.py) -> the transformed slab + the tile's (rm, c0) coords.
    slab, rm, c0 = load_transform(
        tl.program_id(0),
        x_ptr, t_ptr,
        M, C, GT,
        stride_xm, stride_xk,
        stride_tc, stride_ti, stride_tj,
        TILE_M, WIDTH,
        NT, USE_DOT, ACC,
    )

    # quantize each length-GQ group: the slab's WIDTH cols split into NQ = WIDTH//GQ groups (each transform block GT into GT//GQ) -> scaled values + scales.
    xs, scale_q, scale_q8 = quant_group(slab.reshape(TILE_M, NQ, GQ), gs, ss, s_ptr.dtype.element_ty, ACC)
    # encode the scaled values to e2m1. When packing, the byte is the single source for the value + packed
    # stores; dequant-only skips the byte (round_value) so the libdevice path avoids a pack/unpack round-trip.
    # (The e2m1 helpers take the whole (TILE_M, NQ, GQ) tile and do any pair split/join internally on the cvt path.)
    if DO_PACK:
        byte = round_pack(xs, USE_HW)  # (TILE_M, NQ, GQ // 2)
        if DO_DEQUANT:
            val = dequant(byte, USE_HW)  # (TILE_M, NQ, GQ), consistent with the packed byte
    elif DO_DEQUANT:
        val = round_value(xs, USE_HW)  # (TILE_M, NQ, GQ) fake-quant only: no byte

    # store this program's swizzled TILE_M-row tile (rows rm, columns c0*GT .. +WIDTH; NQ quant groups within).
    # whole columns are valid/invalid together, so a contiguous column-range mask is exactly the group mask.
    mask_m = rm < M
    if DO_DEQUANT:
        # fake-quant values, in the same wide column layout block_transform writes y.
        col = c0 * GT + tl.arange(0, WIDTH)  # the tile's WIDTH (= NT * GT = NQ * GQ) contiguous columns
        mask_col = col < C * GT
        y = (val * (scale_q * gs)).reshape(TILE_M, WIDTH)  # grid * (scale_q * gs)
        tl.store(
            y_ptr + rm[:, None] * stride_ym + col[None, :] * stride_yk,
            y,
            mask=mask_m[:, None] & mask_col[None, :],
        )  # auto-casts
    if DO_PACK:
        cc = c0 * GT // GQ + tl.arange(0, NQ)  # the NQ quant (= scale) groups this tile owns
        mask_cc = cc < C * GT // GQ  # total quant groups = K // GQ
        # --- sm_89 fp8-store codegen bug + fix (this scale store) ---
        # BUG: on sm_89 a Triton convert_layout codegen bug corrupts ~half the rows of this store when
        #   scale_q8 is an fp8 (e4m3 / float8e4nv) value derived from the tl.dot above AND the tile spans >1
        #   MMA M-subtile (TILE_M >= 32). It binds to the fp8 SSA value, not the store dtype -- bitcasting fp8
        #   -> uint8 right before the store does NOT help; only never creating the fp8 value does.
        # FIX (used here): on sm8x quant_group returns scale_q8 as a plain int8 byte (xxfp4.to_e4m3 + an
        #   int8-viewed s_ptr, no fp8 value), so this store is integer and bit-exact at EVERY TILE_M.
        # ALT FIX (simpler, if the integer codec ever regresses): pin the autotuner to TILE_M == 16 -- a single
        #   m16 subtile never spans subtiles, so the fp8 store is correct; the cost is losing the wider-tile
        #   configs. sm_90+ has neither problem (native fp8 cvt + store; verified bit-exact at TILE_M=32).
        tl.store(
            s_ptr + rm[:, None] * stride_sm + cc[None, :] * stride_sc,
            scale_q8.reshape(TILE_M, NQ),
            mask=mask_m[:, None] & mask_cc[None, :],
        )
        colq = c0 * GT // 2 + tl.arange(0, WIDTH // 2)  # packed bytes: 2 codes per byte, WIDTH // 2 per tile
        mask_colq = colq < C * (GT // 2)
        packed = byte.reshape(TILE_M, WIDTH // 2)  # NQ * (GQ // 2) = WIDTH // 2
        tl.store(
            q_ptr + rm[:, None] * stride_qm + colq[None, :] * stride_qk,
            packed,
            mask=mask_m[:, None] & mask_colq[None, :],
        )


def transform_quantize(
        x: torch.Tensor,
        transform: torch.Tensor,
        *,
        quant_group_size: int | None = None,
        scale_dtype: torch.dtype = torch.float8_e8m0fnu,
        scale_scale: torch.Tensor | float = 4.,
        global_scale: torch.Tensor | float = 1. / 3.,
        do_dequant: bool = True,
        do_pack: bool = True,
        transposed: bool = False,
        inference_compat: bool = False,
):
    """
    Fused block transform + XXFP4 quantize of x:(...,K). `transform` is a (C, GT, GT) per-block stack or
    a 2D (GT, GT) repeated matrix (see transform.block_transform). quant_group_size (GQ, a power of two in
    [2, 128] dividing K) sets the quant group, decoupled from the transform block size GT: it defaults to GT
    (one block = one group); GQ < GT splits each transformed block into GT//GQ groups (e.g. GT=128, GQ=32 --
    larger mixing, finer quant); GQ > GT makes one group span GQ//GT blocks (e.g. GT=16, GQ=32). GQ > GT
    requires a per-block (C>1) transform -- the repeated 2D / single-block (C==1) layout flattens one GT-block
    per kernel row, so a cross-block group cannot form there. The group size sets the format: 32 for mxfp4, 16
    for nvfp4. Quant fields match xxfp4.quantize (defaults = mxfp4): scale_dtype (float8_e8m0fnu => e8m0, else
    a hardware float8), scale_scale (ss), global_scale (gs); inference_compat shifts the stored e8m0 byte
    (pack only). Returns fake-quant (dequant only), (packed, scale) (pack only), or all three.
    Any decoupled GQ stays fully fused at ALL TILE_M configs on every device: e8m0 and e4m3 both build the
    scale byte with integer ops and store it without an fp8 value (see the module docstring), so the sm_89
    fp8-store convert_layout bug never triggers. GQ == GT (the usual case) is always fully fused.
    """
    assert do_dequant or do_pack, 'select at least one of dequant / pack'
    K: int = x.size(-1)
    if transform.dim() == 2:  # repeated: kernel row = one block (C=1, row width GT)
        GT, GT2 = transform.shape
        C = 1
        assert K % GT == 0, 'last dim must be divisible by GT'
    else:  # per-block: kernel row = the whole K (= C*GT)
        C, GT, GT2 = transform.shape
        assert K == C * GT, 'transform must have K/GT blocks'
    assert GT == GT2 and 2 <= GT <= 128 and (GT & (GT - 1)) == 0 and K % GT == 0, 'GT must be a power of two in [2, 128] dividing K'
    GQ: int = GT if quant_group_size is None else quant_group_size  # quant group size (decoupled from the transform block GT)
    assert 2 <= GQ <= 128 and (GQ & (GQ - 1)) == 0 and K % GQ == 0, 'quant_group_size must be a power of two in [2, 128] dividing K'
    # GQ may be SMALLER than GT (each block splits into GT//GQ groups) OR LARGER (a group spans GQ//GT blocks).
    # GQ > GT needs multi-block kernel rows -- a per-block (C>1) transform, where each row is the full K so the
    # cross-block group lives within one row (then a tile of WIDTH >= GQ columns, with WIDTH % GQ == 0, holds
    # whole groups; the stock widths {128,256,512} satisfy that for any power-of-two GQ <= 128). The repeated
    # 2D / single-block (C==1) layout flattens one GT-block per row, so a GQ>GT group would cross independent
    # rows -- forbidden there. (For GQ <= GT every C works; GT % GQ == 0 is automatic for powers of two.)
    assert GQ <= GT or C > 1, 'GQ > GT (quant group larger than the transform block) requires a per-block (C>1) transform, not a repeated 2D / single-block matrix'
    assert x.dtype == transform.dtype, 'x and transform must share dtype'
    assert transform.is_contiguous(), 'transform must be a contiguous (C, GT, GT) or (GT, GT) tensor'
    e8m0: bool = scale_dtype == torch.float8_e8m0fnu
    # e4m3 (NVFP4) scale: sm_90+ uses the native fp8 cvt + fp8 store; cc<9 (sm8x) passes the scale as an int8
    # view (as_triton_fp8_view, below) and builds/stores the byte with integer ops -- no tl.float8e4nv, so it
    # compiles on sm86 and the in-kernel fp8-store convert_layout bug never triggers; ALL TILE_M configs run.
    device: torch.device = x.device
    sti, stj = (GT, 1) if transposed else (1, GT)
    xc: torch.Tensor = x.contiguous()
    row: int = C * GT  # kernel's row width: K (per-block) or GT (repeated)
    M: int = xc.numel() // row
    lead, n_groups = x.shape[:-1], K // GQ  # n_groups = quant (scale) groups per row = K // GQ
    y: torch.Tensor = torch.empty_like(xc) if do_dequant else xc
    q: torch.Tensor = torch.empty(*lead, K // 2, dtype=torch.float4_e2m1fn_x2, device=device) if do_pack else torch.empty(0, dtype=torch.float4_e2m1fn_x2, device=device)
    s: torch.Tensor = torch.empty(*lead, n_groups, dtype=scale_dtype, device=device) if do_pack else torch.empty(0, dtype=scale_dtype, device=device)
    hw_cvt: bool = has_capability(min_major=10, device_index=device.index)  # Blackwell sm_100+: emit the native e2m1 cvt
    grid = lambda meta: (triton.cdiv(M, meta['TILE_M']) * triton.cdiv(C, meta['WIDTH'] // GT),)
    with device_guard(device):
        _kernel[grid](
            xc, transform, y, q.view(dtype=torch.uint8), as_triton_fp8_view(s),  # e4m3 scale auto-viewed as int8 on sm8x (no tl.float8e4nv); fp8 on sm_90+
            torch.as_tensor(global_scale, dtype=torch.float64, device=device),
            torch.as_tensor(scale_scale, dtype=torch.float64, device=device),
            M=M, C=C, GT=GT, GQ=GQ,
            stride_xm=row, stride_xk=1,
            stride_tc=GT * GT, stride_ti=sti, stride_tj=stj,
            stride_ym=row, stride_yk=1,
            stride_qm=row // 2, stride_qk=1,
            stride_sm=C * GT // GQ, stride_sc=1,  # scales per kernel row = C*GT//GQ (= K//GQ per-block, GT//GQ repeated)
            ISIZE=x.element_size(), DO_DEQUANT=do_dequant, DO_PACK=do_pack, HW_CVT=hw_cvt,
            enable_reflect_ftz=False,  # IEEE fp32 subnormals (no flush-to-zero) so denormal-scale / inf-saturating groups match the reference
        )
    if do_pack:
        s_out: torch.Tensor = compat_shift(s, -2) if inference_compat and e8m0 else s
        return (y, q, s_out) if do_dequant else (q, s_out)
    return y


def transform_fakequant(x, transform, **kwargs):
    """
    Fused transform + XXFP4 fake-quant -> (...,K) (input dtype). See transform_quantize for kwargs.
    """
    return transform_quantize(x=x, transform=transform, do_dequant=True, do_pack=False, **kwargs)


def transform_quantize_pack(x, transform, **kwargs):
    """
    Fused transform + XXFP4 quant + pack -> (packed float4_e2m1fn_x2 (...,K//2), scale (...,K//GQ)).
    """
    return transform_quantize(x=x, transform=transform, do_dequant=False, do_pack=True, **kwargs)


# =========================================================================== #
# Unit test + benchmark  (run from the source root:  CUDA_VISIBLE_DEVICES=<gpu> python -m triton_kernels.fused)
# =========================================================================== #
def _unit_test(
        device: torch.device = torch.device('cuda'),
) -> None:
    """
    Unit test vs the PyTorch reference quantize_fp4.rtn_xxfp4, all BIT-EXACT. With an identity transform
    the fused op reduces to the reference MXFP4 / NVFP4 quant of x (exact codes + scale). With a real
    per-block transform, fusing equals the two steps done separately: it bit-matches rtn_xxfp4 applied to
    the kernel's OWN transform output (transform.block_transform) -- the fused fp32 matmul reduces in a
    different order than einsum, so we compare against the kernel transform (validated vs fp64 in
    transform.py), not an independent one. Also covers DECOUPLED sizes: a transform block GT larger than
    the quant group GQ (GT % GQ == 0), e.g. transform 128 + quantize 32, vs rtn_xxfp4(group_size=GQ).
    Always self-consistent too: pack -> dequantize == fake-quant.
    """

    from quantize_fp4 import rtn_xxfp4, dequant_xxfp4

    torch.manual_seed(seed=0)
    dq = functools.partial(dequant_xxfp4, dtype=torch.bfloat16, high_dtype=torch.float32)
    G: int = 32
    cfg: dict = {'scale_dtype': torch.float8_e8m0fnu, 'scale_scale': 4., 'global_scale': 1. / 3.}  # fused kwargs (G from transform)
    # identity transform: fused == the reference MXFP4 quant of x (identity dot returns x exactly in fp32)
    for M, K in (1000, 4096), (512, 8192):
        x: torch.Tensor = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 2.5  # (M, K)
        identity: torch.Tensor = torch.eye(G, dtype=torch.bfloat16, device=device).unsqueeze(0).repeat(K // G, 1, 1).contiguous()  # (C, G, G)
        fake_quant: torch.Tensor = transform_fakequant(x=x, transform=identity, **cfg)
        packed, scale = transform_quantize_pack(x=x, transform=identity, **cfg)
        reference: dict = rtn_xxfp4(x=x, group_size=G, fp4_rounding_mode='even', high_dtype=torch.float32, round_dtype=torch.bfloat16, **cfg)
        assert fake_quant.equal(reference['fake_quant']), ('identity', M, K)  # e2m1 codes match the reference exactly
        assert scale.view(dtype=torch.uint8).equal(reference['scale_quant'].view(dtype=torch.uint8))  # e8m0 scale matches exactly
        assert dq(e2m1=packed, scale_quant=scale, global_scale=cfg['global_scale']).equal(fake_quant)  # pack -> dequantize round-trips match fake-quant exactly

    # real per-block transform: fusing must equal the two steps done separately. The fused fp32 tensor-core
    # matmul reduces in a different order than einsum, so we quantize the KERNEL's own transform output
    # (block_transform, the same load_transform fused uses) -> fused == that composition, BIT-EXACT. (The
    # transform itself is validated against the fp64 reference in transform.py, and the quant in xxfp4.py.)
    M, K = 1000, 4096
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    transform = torch.randn(K // G, G, G, dtype=torch.bfloat16, device=device) * G ** -.5
    transformed: torch.Tensor = block_transform(x=x, transform=transform, out_dtype=torch.float32)  # the kernel's fp32 transform
    reference_fq: torch.Tensor = rtn_xxfp4(x=transformed, group_size=G, fp4_rounding_mode='even', high_dtype=torch.float32, round_dtype=torch.bfloat16, **cfg)['fake_quant']
    assert transform_fakequant(x=x, transform=transform, **cfg).equal(reference_fq), 'real-transform'  # fused == block_transform then quant, bit-exact
    assert dq(*transform_quantize_pack(x=x, transform=transform, **cfg), global_scale=cfg['global_scale']).equal(transform_fakequant(x=x, transform=transform, **cfg))  # roundtrip exact

    # repeated (shared 2D) transform == the per-block path with the matrix repeated C times (bit-exact)
    shared: torch.Tensor = torch.randn(G, G, dtype=torch.bfloat16, device=device) * G ** -.5
    per_block: torch.Tensor = shared.unsqueeze(0).repeat(K // G, 1, 1).contiguous()
    assert transform_fakequant(x=x, transform=shared, **cfg).equal(transform_fakequant(x=x, transform=per_block, **cfg))

    # NVFP4 (e4m3, G=16), identity transform
    cfg16: dict = {'scale_dtype': torch.float8_e4m3fn, 'scale_scale': 6., 'global_scale': .1}
    G16: int = 16
    x16: torch.Tensor = torch.randn(512, 64 * G16, dtype=torch.bfloat16, device=device) * 2.5
    identity16: torch.Tensor = torch.eye(G16, dtype=torch.bfloat16, device=device).unsqueeze(0).repeat(64, 1, 1).contiguous()
    fake_quant16: torch.Tensor = transform_fakequant(x=x16, transform=identity16, **cfg16)
    packed16, scale16 = transform_quantize_pack(x=x16, transform=identity16, **cfg16)
    reference16: dict = rtn_xxfp4(x=x16, group_size=G16, fp4_rounding_mode='even', high_dtype=torch.float32, round_dtype=torch.bfloat16, **cfg16)
    assert fake_quant16.equal(reference16['fake_quant']), 'nvfp4'  # e2m1 codes match the reference exactly
    assert scale16.view(dtype=torch.uint8).equal(reference16['scale_quant'].view(dtype=torch.uint8))  # e4m3 scale matches exactly
    assert dq(e2m1=packed16, scale_quant=scale16, global_scale=cfg16['global_scale']).equal(fake_quant16)  # pack -> dequantize round-trips matches exactly

    # DECOUPLED transform block (GT) vs quant group (GQ != GT): GQ < GT splits each transformed block into
    # GT//GQ groups (transform 128, quantize 32); GQ > GT makes one group span GQ//GT blocks (transform 16,
    # quantize 32) -- valid for a per-block (C>1) transform. Either way fused == rtn_xxfp4(group_size=GQ) on the
    # kernel's OWN transform output, BIT-EXACT. mxfp4 (e8m0) and nvfp4 (e4m3, GQ=16) both run fully fused over
    # ALL TILE_M configs (the integer e4m3 byte + uint8 store dodge the sm_89 fp8-store bug).
    for GT, GQ in (128, 32), (64, 32), (128, 16), (64, 16), (32, 16), (16, 32), (32, 64):
        M, K = 1000, 4096  # M not a multiple of TILE_M -> exercises the decoupled scale-store's row masking
        xd: torch.Tensor = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        cfgd: dict = ({'scale_dtype': torch.float8_e8m0fnu, 'scale_scale': 4., 'global_scale': 1. / 3.} if GQ != 16
                      else {'scale_dtype': torch.float8_e4m3fn, 'scale_scale': 6., 'global_scale': .1})  # GQ=16 -> nvfp4
        gsd: float = cfgd['global_scale']
        ref_args: dict = {'group_size': GQ, 'fp4_rounding_mode': 'even', 'high_dtype': torch.float32, 'round_dtype': torch.bfloat16}
        # identity transform at GT -> fused reduces to the reference quant of x at group GQ (block size irrelevant)
        ident: torch.Tensor = torch.eye(GT, dtype=torch.bfloat16, device=device).unsqueeze(0).repeat(K // GT, 1, 1).contiguous()
        fq_i: torch.Tensor = transform_fakequant(x=xd, transform=ident, quant_group_size=GQ, **cfgd)
        pk_i, sc_i = transform_quantize_pack(x=xd, transform=ident, quant_group_size=GQ, **cfgd)
        ref_i: dict = rtn_xxfp4(x=xd, **ref_args, **cfgd)
        assert fq_i.equal(ref_i['fake_quant']), ('decoupled-identity', GT, GQ)
        assert sc_i.view(dtype=torch.uint8).equal(ref_i['scale_quant'].view(dtype=torch.uint8)), ('decoupled-identity-scale', GT, GQ)
        assert dq(e2m1=pk_i, scale_quant=sc_i, global_scale=gsd).equal(fq_i), ('decoupled-identity-roundtrip', GT, GQ)
        # real transform at GT -> fused == rtn_xxfp4(group_size=GQ) of the kernel's OWN block_transform output
        td: torch.Tensor = torch.randn(K // GT, GT, GT, dtype=torch.bfloat16, device=device) * GT ** -.5
        transformed_d: torch.Tensor = block_transform(x=xd, transform=td, out_dtype=torch.float32)
        ref_r: dict = rtn_xxfp4(x=transformed_d, **ref_args, **cfgd)
        fq_r: torch.Tensor = transform_fakequant(x=xd, transform=td, quant_group_size=GQ, **cfgd)
        pk_r, sc_r = transform_quantize_pack(x=xd, transform=td, quant_group_size=GQ, **cfgd)
        assert fq_r.equal(ref_r['fake_quant']), ('decoupled-real', GT, GQ)
        assert sc_r.view(dtype=torch.uint8).equal(ref_r['scale_quant'].view(dtype=torch.uint8)), ('decoupled-real-scale', GT, GQ)
        assert dq(e2m1=pk_r, scale_quant=sc_r, global_scale=gsd).equal(fq_r), ('decoupled-real-roundtrip', GT, GQ)
        # repeated (shared 2D, C==1) transform == the per-block path (also exercises the C==1 pack/scale store);
        # only for GQ <= GT -- the C==1 layout (one GT-block per kernel row) cannot form a GQ > GT cross-block group.
        if GQ <= GT:
            shared_d: torch.Tensor = td[0].contiguous()
            per_block_d: torch.Tensor = shared_d.unsqueeze(0).repeat(K // GT, 1, 1).contiguous()
            pk_s, sc_s = transform_quantize_pack(x=xd, transform=shared_d, quant_group_size=GQ, **cfgd)
            pk_p, sc_p = transform_quantize_pack(x=xd, transform=per_block_d, quant_group_size=GQ, **cfgd)
            assert pk_s.view(dtype=torch.uint8).equal(pk_p.view(dtype=torch.uint8)) and sc_s.view(dtype=torch.uint8).equal(sc_p.view(dtype=torch.uint8)), ('decoupled-repeated', GT, GQ)

    # inference_compat (e8m0): the stored exponent byte shifts -2, dequantize restores it -> default fake-quant
    identity_x: torch.Tensor = torch.eye(G, dtype=torch.bfloat16, device=device).unsqueeze(0).repeat(K // G, 1, 1).contiguous()  # matches x (M, K)
    packed_c, scale_c = transform_quantize_pack(x=x, transform=identity_x, inference_compat=True, **cfg)
    scale_c_restored: torch.Tensor = (scale_c.view(dtype=torch.uint8).to(dtype=torch.int16) + 2).clamp(min=0, max=255).to(dtype=torch.uint8).view(dtype=torch.float8_e8m0fnu)  # undo inference_compat -2 e8m0 exponent shift (+2 byte == x4)
    assert dq(e2m1=packed_c, scale_quant=scale_c_restored, global_scale=cfg['global_scale']).equal(transform_fakequant(x=x, transform=identity_x, **cfg))

    print('Unit test passed.')


def _benchmark(
        device: torch.device = torch.device('cuda'),
) -> None:
    """
    Throughput (input GB/s) of the fused transform + quant vs the two-step PyTorch reference
    (apply_block_transform then rtn_xxfp4).
    """

    from block_transform import apply_block_transform
    from quantize_fp4 import rtn_xxfp4

    G: int = 32

    def torch_reference(x, transform):  # apply_block_transform then MXFP4 quant + pack
        return rtn_xxfp4(x=apply_block_transform(x=x, transform=transform), group_size=G, scale_dtype=torch.float8_e8m0fnu, scale_scale=4., global_scale=1. / 3., fp4_rounding_mode='even')

    print(f"{'K':>8} | {'fused_fakeq':>12} | {'fused_pack':>12} | {'torch_2step':>12}    (input GB/s)")
    for K in (2 ** i for i in range(10, 16)):
        x: torch.Tensor = torch.randn(4096, K, dtype=torch.bfloat16, device=device)
        transform: torch.Tensor = torch.randn(K // G, G, G, dtype=torch.bfloat16, device=device) * G ** -.5
        gbps = lambda fn: x.numel() * x.element_size() * 1e-9 / (triton.testing.do_bench(fn) * 1e-3)  # bytes of x read, GB/s
        print(f'{K:>8} | {gbps(lambda: transform_fakequant(x=x, transform=transform)):>12.1f} | {gbps(lambda: transform_quantize_pack(x=x, transform=transform)):>12.1f} | {gbps(lambda: torch_reference(x, transform)):>12.1f}')


if __name__ == '__main__':
    _unit_test(device=torch.device('cuda'))
    _benchmark(device=torch.device('cuda'))
