"""
XXFP4: unified MXFP4 / NVFP4 group quantization (E2M1 codes + per-group scale).

Triton port of `rtn_xxfp4`, round-to-nearest-EVEN onto the e2m1 grid {0,+-.5,+-1,+-1.5,+-2,+-3,+-4,+-6} (clamp |.|<=6).
One algorithm, parameterized directly by group_size, scale_dtype (torch.float8_e8m0fnu => e8m0, or any
hardware-castable float8 -- torch.float8_e4m3fn / e5m2 / ...), global_scale (gs) and scale_scale (ss).

Per group of group_size:
    amax = max|x|;   scale = amax / (ss * gs);   scale_q = quant(scale -> scale_dtype)
    x_scaled = x / (scale_q * gs);   code = RNE(x_scaled);   dequant = grid[code] * (scale_q * gs)
An existing_scale may be supplied to skip the amax step and round x against a precomputed scale.

Common configs (pass the fields directly; the defaults are mxfp4):
    mxfp4: group_size=32, scale_dtype=torch.float8_e8m0fnu, global_scale=1/3, scale_scale=4
    nvfp4: group_size=16, scale_dtype=torch.float8_e4m3fn,  global_scale=0.1, scale_scale=6

inference_compat (e8m0 only, Python-side): store the MX shared exponent 2^(f-2) e8m0 (byte f-2+127,
    bit-compatible with an external fp4 e8m0 storage convention) instead of the default nearest-e8m0 2^f (byte f+127).
    e8m0 stores log2, so this is just a byte offset of -2 on store (scale * 0.25) and +2 on read (scale * 4)
    -- the e2m1 codes and fake-quant *values* are identical, only the stored exponent differs.
    The kernel always emits / consumes the default; quantize() / dequantize() apply the shift.
"""

import functools

import torch
import triton
import triton.language as tl
from triton.language.extra.libdevice import div_rn  # IEEE round-to-nearest fp division (Triton's default / is an approx reciprocal)

from triton_kernels.e2m1 import round_pack, dequant, round_value, ELT_CONFIGS
from triton_kernels.guard import device_guard, has_capability


@functools.lru_cache(maxsize=None)
def fp8_native_rne(device_index: int) -> bool:
    """
    Can this device use the native Triton fp32 -> e4m3 cast for the scale? True on sm_90+ (the Hopper /
    Blackwell hardware cvt exists and is RNE -- verified bit-exact on sm_120 over the full e4m3 grid incl.
    midpoints / subnormals / +-448 saturation). False on cc < 9 (sm8x): on sm_89 (Ada) the emulated cast
    truncates toward zero AND a tl.dot-fed fp8 scale store miscompiles, and on sm86 fp8e4nv is unsupported
    entirely -- so there as_triton_fp8_view (which queries this) hands the e4m3 scale to the kernel as an int8
    view and the byte is built / decoded with the pure-integer to_e4m3 / from_e4m3 (no tl.float8e4nv anywhere).
    """
    return has_capability(min_major=9, device_index=device_index)


# =========================================================================== #
# Scale-cast shims (pure integer / fp, NO fp8 value). Triton has no e8m0 type, and on cc < 9 (sm8x) the e4m3
# scale also avoids tl.float8e4nv: on sm86 fp8e4nv is unsupported, and on sm_89 the emulated cast truncates
# (not RNE) AND a tl.dot-fed fp8 store miscompiles (see fused.py). So there the scale byte is built with
# integer ops -- to_e8m0 / to_e4m3 -> a byte -- and decoded by from_e8m0 / from_e4m3; the scale is
# handed to the kernel as a uint8 (e8m0) / int8 (e4m3) view (as_triton_fp8_view), so a plain tl.store matches
# its pointer. sm_90+ (fp8_native_rne) instead uses the one-line hardware e4m3 cvt (RNE) + native fp8 store.
# =========================================================================== #
@triton.jit
def to_e8m0(v):
    """
    fp value (>= 0) -> e8m0 byte (uint8). Round-to-nearest power of two (linear midpoint 1.5*2^e, tie up),
    correct across the whole range including subnormals; zero / underflow -> byte 0, overflow / inf -> byte
    254 (saturate to the e8m0 max 2^127), NaN -> byte 255. Matches the rtn_xxfp4 reference; bit-exact vs
    torch on normal v (torch rounds the subnormal regime differently -- we keep true nearest there, and a
    scale never reaches it).

    Pure-integer rounding (no fp arithmetic): for normal fp32 v = 1.f * 2^(E-127), the nearest power of two
    rounds UP iff 1.f >= 1.5 iff the mantissa MSB (bit 22) is set, so adding 1<<22 carries that bit into the
    exponent field -- ((bits + 0x400000) >> 23) & 0xFF is the round-half-up e8m0 byte (saturated to 254).
    Subnormals (exp field 0) have no implicit 1, so a guard does the nearest there directly: byte 1 iff
    v >= 1.5*2^-127 (mantissa bits >= 0x600000), else byte 0 (the e8m0 minimum 2^-127; v == 0 -> 0).
    """
    vf = v.to(tl.float32)  # bit ops need a 32-bit float; fp64 inputs exist (ref uses high_dtype=fp32)
    bits = vf.to(tl.uint32, bitcast=True)
    norm = tl.minimum(((bits + 0x00400000) >> 23) & 0xFF, 254)  # round-half-up exponent (mantissa-MSB carry); saturate inf/overflow to e8m0 max 254
    sub = ((bits >> 23) & 0xFF) == 0  # subnormal (or zero): exp field 0, no implicit 1
    byte = tl.where(sub, (bits >= 0x00600000).to(tl.uint32), norm)  # subnormal: nearest -> byte 1 iff v >= 1.5*2^-127, else 0
    return tl.where(vf != vf, 255, byte).to(tl.uint8)  # NaN -> 255


@triton.jit
def from_e8m0(scale_q8, ACC: tl.constexpr):
    """
    e8m0 byte -> fp value 2^(byte-127) in dtype ACC, == casting a float8_e8m0fnu back to float; byte 255
    (the e8m0 NaN) -> NaN. Native equivalent: `scale_q8.bitcast(tl.float8_e8m0fnu).to(ACC)`.
    """
    return tl.where(scale_q8 == 255, float('nan'), (scale_q8.to(ACC) - 127.).exp2())  # byte 255 = e8m0 NaN


@triton.jit
def _e4m3_decode_mag(byte):
    """
    e4m3 magnitude byte (int8; sign bit 7 ignored) -> fp32 magnitude: normal (exp field ef >= 1) is
    2^(ef-7) * (1 + m/8), subnormal (ef == 0) is 2^-6 * (m/8), the 0x7f slot is NaN. fp arithmetic, no fp8
    value -- shared by to_e4m3 (neighbor distances) and from_e4m3 (decode).
    """
    b = byte.to(tl.int32) & 0x7F
    ef = (b >> 3) & 0xF
    m = (b & 0x7).to(tl.float32)
    normal = (1. + m * .125) * (ef - 7).to(tl.float32).exp2()
    sub = m * .001953125  # 2^-6 * m/8
    return tl.where(b == 0x7F, float('nan'), tl.where(ef >= 1, normal, sub))


@triton.jit
def _e4m3_trunc_byte(af):
    """
    fp32 magnitude (af >= 0) -> toward-zero e4m3 magnitude byte (int8), saturating to 0x7e (448). Pure
    integer: for normals (exp field e4 = (E-127)+7 in [1,15]) drop the low 20 mantissa bits; for subnormals
    (e4 <= 0) shift the 24-bit significand right by 14-e (= floor(af / 2^-9)); overflow (e4 > 15) and the
    0x7f NaN slot saturate to 0x7e. The RNE choice is made by to_e4m3 against this byte's +1 neighbor.
    """
    mag = af.to(tl.int32, bitcast=True) & 0x7FFFFFFF
    e = (mag >> 23).to(tl.int32) - 127
    m = (mag & 0x7FFFFF).to(tl.int32)
    e4 = e + 7
    m3 = m >> 20
    byte = tl.where(e4 >= 1, ((e4 & 0xF) << 3) | m3, ((1 << 23) | m) >> (14 - e))
    byte = tl.where((e4 > 15) | ((e4 == 15) & (m3 == 7)), 0x7e, byte)  # > 448 and the 0x7f NaN slot saturate
    return byte.to(tl.int8)


@triton.jit
def to_e4m3(v):
    """
    fp value -> e4m3fn BYTE (int8), round-to-nearest-even, == `v.to(torch.float8_e4m3fn)` (saturating to
    +-448, matching rtn_xxfp4's clamp-to-max), NaN -> 0x7f. PURE integer / fp -- NO tl.float8e4nv op is ever
    emitted, so the byte stores into an int8-viewed scale (sm8x) and the kernel compiles where fp8e4nv is
    unsupported (sm86) / miscompiles the fp8 store (sm_89, see fused.py). Rounds by comparing the toward-zero
    byte (_e4m3_trunc_byte) with its next-magnitude neighbor (byte + 1), robust across subnormals and the
    +-448 max; decode with from_e4m3. Native equivalent (sm_90+ hardware cvt is already RNE): `v.to(tl.float8e4nv)`.
    """
    vf = v.to(tl.float32)
    sign = (vf.to(tl.int32, bitcast=True) >> 31).to(tl.int8) << 7
    af = vf.abs()
    lo = _e4m3_trunc_byte(af)  # toward-zero magnitude byte
    up = lo + 1  # next magnitude up (0x7e -> 0x7f is the NaN slot, skipped below)
    lo_f, up_f = _e4m3_decode_mag(lo), _e4m3_decode_mag(up)
    dlo, dup = (af - lo_f).abs(), (af - up_f).abs()
    pick_up = (up_f == up_f) & ((dup < dlo) | ((dup == dlo) & ((lo & 1) == 1)))  # nearer; tie -> even; skip NaN at the top
    byte = tl.where(pick_up, up, lo) | sign
    return tl.where(vf != vf, tl.full(af.shape, 0x7f, tl.int8), byte)  # NaN -> 0x7f


@triton.jit
def from_e4m3(byte, ACC: tl.constexpr):
    """
    e4m3fn byte (int8) -> fp value (dtype ACC), == casting float8_e4m3fn back to float: sign-honored
    magnitude (_e4m3_decode_mag), 0x7f/0xff -> NaN, 0x00 -> +0.0. Pure fp, no fp8 value -- used for the
    in-kernel code computation on sm_89; decode_scale keeps the exact native f8 -> fp cast for the load path.
    """
    mag = _e4m3_decode_mag(byte).to(ACC)
    val = mag * tl.where((byte >> 7) != 0, -1., 1.)  # signed-zero-safe: the -0 byte (0x80) decodes to -0.0, bit-matching the hw cvt decode (int8 sign bit: byte>>7 is 0 or -1; -mag would give +0 -- triton lowers it as 0-0)
    return tl.where((byte & 0x7F) == 0x7F, float('nan'), val)


# =========================================================================== #
# Composable group-quant device functions. The COMPUTE (per-group scale + e2m1 codes) is shared by
# _quant_kernel and fused._kernel; the STORES are not -- each kernel addresses its own memory layout
# (this module: flat consecutive groups; fused: a swizzled 2D tile) with a few local tl.store calls,
# which reads clearer than threading precomputed offset / mask tensors through a shared helper.
# =========================================================================== #
@triton.jit
def decode_scale(scale_q8, ACC: tl.constexpr):
    """
    Decode a quantized group scale to fp (dtype ACC): a uint8 e8m0 byte via from_e8m0; an int8-viewed e4m3
    byte (sm8x, where the scale is passed as int8 so the kernel needs no tl.float8e4nv -- sm86-safe) via
    from_e4m3; else a hardware float8 via fp8 -> fp32 -> ACC (e4m3 on sm_90+, generic float8). Callers round x
    with div_rn(x, scale_q * gs) and dequantize with grid * (scale_q * gs), matching the reference's grouped ops.
    """
    if scale_q8.dtype == tl.uint8:  # uint8 scale (an e8m0 view) => e8m0 path
        return from_e8m0(scale_q8, ACC)
    elif scale_q8.dtype == tl.int8:  # int8 scale (an e4m3 view on sm8x) => integer e4m3 decode (no fp8 cast)
        return from_e4m3(scale_q8, ACC)
    else:
        return scale_q8.to(tl.float32).to(ACC)  # hardware float8 -> fp (e4m3 sm_90+, generic); no direct fp8 -> fp64 cast


@triton.jit
def _amax_nan(a, b):  # NaN-propagating max combine for tl.reduce (the tl.max reduction silently drops NaN)
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def quant_group(x, gs, ss, S_DTYPE: tl.constexpr, ACC: tl.constexpr):
    """
    x:(...,G) fp -> (xs (...,G), scale_q (...,1), scale_q8 (...,1)). Per-group AbsMax scale
    (over the LAST axis), quantized + decoded, dispatched on the scale view dtype S_DTYPE: tl.uint8 => e8m0
    byte (mxfp4); tl.int8 => an int8-viewed e4m3 scale (sm8x), built with the pure-integer to_e4m3 BYTE and
    decoded with from_e4m3 -- NO tl.float8e4nv op, so the kernel compiles on sm86 (where fp8e4nv is unsupported)
    and dodges the sm_89 fp8-store bug; tl.float8e4nv => e4m3 via the native hardware cvt on sm_90+ (RNE); else
    a generic hardware float8 cast. scale_q8 matches S_DTYPE (int8 for the sm8x e4m3 byte, fp8/uint8 otherwise),
    so the caller's plain tl.store needs no reinterpret. gs / ss arrive already in ACC (the caller loads the
    0-d scalar tensors and casts). Returns the SCALED values xs = x/(scale_q*gs) (the caller encodes them to
    e2m1 via round_pack / round_value) and the DECODED scale_q; the caller dequantizes via grid * (scale_q * gs).
    """
    amax = x.abs().reduce(axis=-1, combine_fn=_amax_nan, keep_dims=True)  # NaN-propagating AbsMax (the tl.max reduction drops NaN, unlike torch.amax)
    v = div_rn(amax, ss * gs)  # AbsMax scale amax/(ss*gs), IEEE division in the reference's op order
    if S_DTYPE == tl.uint8:  # e8m0 (mxfp4): integer byte; from_e8m0 yields NaN for byte 255 -> codes/dequant NaN-saturate
        scale_q8 = to_e8m0(v)
        scale_q = from_e8m0(scale_q8, ACC)
    elif S_DTYPE == tl.int8:  # e4m3 (nvfp4) on sm8x: int8-viewed scale -> pure-integer byte, NO tl.float8e4nv (sm86-safe)
        scale_q8 = to_e4m3(v)
        scale_q = from_e4m3(scale_q8, ACC)
    else:
        # e4m3 (nvfp4) on sm_90+: native RNE hardware cvt -> fp8 value + fp8 store
        # generic hardware float8 (e5m2, ...) -- easy fallback
        scale_q8 = v.to(tl.float32).to(S_DTYPE)
        scale_q = scale_q8.to(tl.float32).to(ACC)  # decode (upcast, exact)
    xs = div_rn(x, scale_q * gs)  # scaled values x/(scale_q*gs); shared across all scale dtypes (caller encodes)
    return xs, scale_q, scale_q8  # decoded scale; caller dequantizes via grid * (scale_q * gs)


# =========================================================================== #
# Elementwise group kernels (each program owns BLOCK_R = BLOCK // G whole groups)
# =========================================================================== #
@triton.autotune(configs=ELT_CONFIGS, key=['G', 'HAS_SCALE', 'DO_DEQUANT', 'DO_PACK', 'HW_CVT'])
@triton.jit
def _quant_kernel(
        x_ptr, y_ptr, q_ptr, s_ptr, gs_ptr, ss_ptr,
        n_groups,
        G: tl.constexpr,
        HAS_SCALE: tl.constexpr, DO_DEQUANT: tl.constexpr, DO_PACK: tl.constexpr, HW_CVT: tl.constexpr,
        BLOCK: tl.constexpr,
):
    BLOCK_R: tl.constexpr = BLOCK // G  # groups per program
    ACC: tl.constexpr = tl.float64 if x_ptr.dtype.element_ty == tl.float64 else tl.float32  # fp64 in -> fp64 compute
    USE_HW: tl.constexpr = HW_CVT and x_ptr.dtype.element_ty != tl.float64  # the e2m1 cvt is f32-only; fp64 keeps the libdevice round
    gs = tl.load(gs_ptr).to(ACC)  # gs_ptr / ss_ptr -> 0-d fp64 scalar tensors loaded at full precision (a python-float kernel arg would truncate to fp32)
    ss = tl.load(ss_ptr).to(ACC)
    pid = tl.program_id(0)
    g = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    mg = g < n_groups
    c = tl.arange(0, G)
    x = tl.load(x_ptr + g[:, None] * G + c[None, :], mask=mg[:, None]).to(ACC)
    # per-group quantized scale: reuse a precomputed one, or compute it from amax (quant_group).
    if HAS_SCALE:
        scale_q8 = tl.load(s_ptr + g[:, None], mask=mg[:, None])
        scale_q = decode_scale(scale_q8, ACC)
        xs = div_rn(x, scale_q * gs)  # x/(scale_q*gs), IEEE division, matching the reference
    else:
        xs, scale_q, scale_q8 = quant_group(x, gs, ss, s_ptr.dtype.element_ty, ACC)
    # encode the scaled values to e2m1. When packing, the byte is the single source for both stores; dequant-only
    # skips the byte (round_value) so the libdevice path avoids a pack/unpack round-trip. (The e2m1 helpers take
    # the whole (BLOCK_R, G) tile and do any pair split/join internally on the cvt path.)
    if DO_PACK:
        byte = round_pack(xs, USE_HW)  # (BLOCK_R, G // 2)
        if DO_DEQUANT:
            val = dequant(byte, USE_HW)  # (BLOCK_R, G), consistent with the packed byte
    elif DO_DEQUANT:
        val = round_value(xs, USE_HW)  # (BLOCK_R, G) fake-quant only: no byte
    # flat group layout: group g owns y elements g*G + [0,G), q bytes g*(G//2) + [0,G//2), scale g.
    if DO_DEQUANT:
        tl.store(y_ptr + g[:, None] * G + c[None, :], val * (scale_q * gs), mask=mg[:, None])  # grid * (scale_q * gs); auto-casts
    if DO_PACK:
        if not HAS_SCALE:
            tl.store(s_ptr + g[:, None], scale_q8, mask=mg[:, None])  # scale_q8 dtype matches s_ptr (int8 e4m3 view on sm8x); an existing scale is returned as-is
        b = pid * BLOCK_R * (G // 2) + tl.arange(0, BLOCK_R * (G // 2))
        tl.store(q_ptr + b, byte.reshape(BLOCK_R * G // 2), mask=b < n_groups * (G // 2))


@triton.autotune(configs=ELT_CONFIGS, key=['G', 'HW_CVT'])
@triton.jit
def _dequant_kernel(
        q_ptr, s_ptr, y_ptr, gs_ptr,
        n_groups,
        G: tl.constexpr,
        HW_CVT: tl.constexpr,
        BLOCK: tl.constexpr,
):
    BLOCK_R: tl.constexpr = BLOCK // G
    ACC: tl.constexpr = tl.float64 if y_ptr.dtype.element_ty == tl.float64 else tl.float32  # fp64 out -> fp64 compute
    USE_HW: tl.constexpr = HW_CVT and y_ptr.dtype.element_ty != tl.float64  # the e2m1 cvt is f32-only; fp64 keeps the libdevice decode
    gs = tl.load(gs_ptr).to(ACC)
    pid = tl.program_id(0)
    g = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    mg = g < n_groups
    b = pid * BLOCK_R * (G // 2) + tl.arange(0, BLOCK_R * (G // 2))
    s = tl.load(s_ptr + g[:, None], mask=mg[:, None])
    scale_q = decode_scale(s, ACC)
    c = tl.arange(0, G)
    val = dequant(tl.load(q_ptr + b, mask=b < n_groups * (G // 2)), USE_HW).reshape(BLOCK_R, G)
    tl.store(y_ptr + g[:, None] * G + c[None, :], val * (scale_q * gs), mask=mg[:, None])  # grid * (scale_q * gs); auto-casts


# =========================================================================== #
# Python API
# =========================================================================== #
def compat_shift(s: torch.Tensor, delta: int) -> torch.Tensor:
    """
    inference_compat e8m0 byte offset (store f-2 / read f): -2 = scale * 0.25, +2 = scale * 4. Preserves dtype.
    """
    return ((s.view(dtype=torch.uint8).to(dtype=torch.int16) + delta).clamp(0, 254).to(dtype=torch.uint8)).view(dtype=s.dtype)


def as_triton_fp8_view(s: torch.Tensor) -> torch.Tensor:
    """
    Hand the kernel a non-float8 view of the scale so the kernel needs no tl.float8e4nv on archs that lack it:
    an e8m0 (float8_e8m0fnu) scale always as uint8; an e4m3 (float8_e4m3fn) scale as int8 on cc < 9 (sm8x,
    where fp8e4nv is unsupported or hits the fp8-store bug -- per fp8_native_rne, queried here on s.device).
    Other float8 scales (and e4m3 on sm_90+) pass through unchanged for the native hardware cvt. The byte
    content is identical either way.
    """
    if s.dtype == torch.float8_e8m0fnu:
        return s.view(dtype=torch.uint8)
    if s.dtype == torch.float8_e4m3fn:
        device_index: int = s.device.index if s.device.index is not None else torch.cuda.current_device()
        if not fp8_native_rne(device_index):  # sm8x: int8 view (no tl.float8e4nv); sm_90+: keep fp8 for the native cvt
            return s.view(dtype=torch.int8)
    return s


def quantize(
        x: torch.Tensor,
        *,
        group_size: int = 32,
        scale_dtype: torch.dtype = torch.float8_e8m0fnu,
        scale_scale: torch.Tensor | float = 4.,
        global_scale: torch.Tensor | float = 1. / 3.,
        existing_scale: torch.Tensor | None = None,
        do_dequant: bool = True,
        do_pack: bool = True,
        inplace: bool = False,
        inference_compat: bool = False,
):
    """
    XXFP4 quantize (...,K), round-to-nearest-even. The defaults are the mxfp4 preset; pass the nvfp4
    fields for nvfp4 (see the module docstring).
      group_size:   elements per scale group (mxfp4 32, nvfp4 16).
      scale_dtype:  per-group scale dtype -- torch.float8_e8m0fnu => e8m0 (manual rounding), else a
                    hardware-castable float8 (torch.float8_e4m3fn / e5m2 / ...).
      scale_scale, global_scale: ss and gs in scale = amax/(ss*gs), dequant = grid*(scale_q*gs).
      existing_scale: (...,K//group_size) precomputed quantized scale (scale_dtype) to round x against,
                    instead of computing it from amax; returned unchanged as the pack scale.
      do_dequant:   produce the fake-quantized (round-tripped) tensor.
      do_pack:      produce packed e2m1 codes float4_e2m1fn_x2 (...,K//2) + per-group scale (...,K//group_size).
      inplace:      write the fake-quant result back into x (requires dequant + contiguous x).
      inference_compat: store / read the MX floor exponent 2^(f-2) e8m0 (bit-compatible with an external
                    fp4 e8m0 convention); e8m0 only, no-op otherwise (fake-quant values are identical).
    Returns fake-quant (dequant only), (packed, scale) (pack only), or (fake-quant, packed, scale).
    """
    assert do_dequant or do_pack, 'select at least one of dequant / pack'
    assert not (inplace and not do_dequant), 'inplace requires dequant=True'
    device: torch.device = x.device
    e8m0: bool = scale_dtype == torch.float8_e8m0fnu
    has_scale: bool = existing_scale is not None
    K: int = x.size(-1)
    assert K % group_size == 0
    if inplace:
        assert x.is_contiguous(), 'inplace requires a contiguous tensor'
        xc: torch.Tensor = x
    else:
        xc: torch.Tensor = x.contiguous()
    n: int = xc.numel() // group_size
    y: torch.Tensor = xc if (inplace or not do_dequant) else torch.empty_like(xc)
    if has_scale:
        # feed the kernel the default (nearest, byte f+127) convention; existing_scale is in whatever
        # representation quantize would output for this inference_compat (compat => stored f-2 => shift +2).
        s: torch.Tensor = compat_shift(existing_scale, +2) if inference_compat and e8m0 else existing_scale
        s: torch.Tensor = s.contiguous()
    elif do_pack:
        s: torch.Tensor = torch.empty(*x.shape[:-1], K // group_size, dtype=scale_dtype, device=device)
    else:
        s: torch.Tensor = torch.empty(0, dtype=scale_dtype, device=device)  # dummy: only carries the dtype
    if do_pack:
        q: torch.Tensor = torch.empty(*x.shape[:-1], K // 2, dtype=torch.float4_e2m1fn_x2, device=device)
    else:
        q: torch.Tensor = torch.empty(0, dtype=torch.float4_e2m1fn_x2, device=device)
    hw_cvt: bool = has_capability(min_major=10, device_index=device.index)  # Blackwell sm_100+: emit the native e2m1 cvt
    with device_guard(device):
        _quant_kernel[lambda m: (triton.cdiv(n, m['BLOCK'] // group_size),)](
            xc, y, q.view(dtype=torch.uint8), as_triton_fp8_view(s),
            torch.as_tensor(global_scale, dtype=torch.float64, device=device),
            torch.as_tensor(scale_scale, dtype=torch.float64, device=device),
            n_groups=n,
            G=group_size,
            HAS_SCALE=has_scale, DO_DEQUANT=do_dequant, DO_PACK=do_pack, HW_CVT=hw_cvt,
            enable_reflect_ftz=False,  # IEEE fp32 subnormals (no flush-to-zero) so denormal-scale / inf-saturating groups match the reference
        )
    if do_pack:
        s_out: torch.Tensor = existing_scale if has_scale else (compat_shift(s, -2) if inference_compat and e8m0 else s)
        if do_dequant:
            return y, q, s_out
        else:
            return q, s_out
    else:
        return y


def fake_quantize(x: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    XXFP4 fake-quant (...,K) -> same shape/dtype. See quantize for kwargs.
    """
    return quantize(x=x, do_dequant=True, do_pack=False, **kwargs)


def quantize_pack(x: torch.Tensor, **kwargs):
    """
    XXFP4 quantize + pack (...,K) -> (packed float4_e2m1fn_x2 (...,K//2), scale (...,K//group_size)).
    """
    return quantize(x=x, do_dequant=False, do_pack=True, **kwargs)


def dequantize(
        packed: torch.Tensor,
        scale: torch.Tensor,
        *,
        global_scale: torch.Tensor | float = 1. / 3.,
        dtype: torch.dtype = torch.bfloat16,
        inference_compat: bool = False,
) -> torch.Tensor:
    """
    Unpack + dequantize. packed float4_e2m1fn_x2 (...,K//2), scale (...,K//group_size) -> (...,K). group_size is inferred from the shapes.
    """
    K: int = packed.size(-1) * 2
    G: int = K // scale.size(-1)  # group_size inferred from the packed / scale shapes
    device: torch.device = packed.device
    e8m0: bool = scale.dtype == torch.float8_e8m0fnu
    pc: torch.Tensor = packed.contiguous()
    sc: torch.Tensor = compat_shift(scale, +2) if (inference_compat and e8m0) else scale  # restore f from the stored f-2
    sc: torch.Tensor = sc.contiguous()
    n: int = sc.numel()
    y: torch.Tensor = torch.empty(*packed.shape[:-1], K, dtype=dtype, device=device)
    hw_cvt: bool = has_capability(min_major=10, device_index=device.index)  # Blackwell sm_100+: emit the native e2m1 cvt
    with device_guard(device):
        _dequant_kernel[lambda m: (triton.cdiv(n, m['BLOCK'] // G),)](
            pc.view(dtype=torch.uint8), as_triton_fp8_view(sc), y,  # FP4 bytes as uint8
            torch.as_tensor(global_scale, dtype=torch.float64, device=device),
            n_groups=n,
            G=G,
            HW_CVT=hw_cvt,
            enable_reflect_ftz=False,  # IEEE fp32 subnormals (no flush-to-zero); matches the reference on denormal-scale groups
        )
    return y


# =========================================================================== #
# Unit test + benchmark  (run from the source root:  CUDA_VISIBLE_DEVICES=<gpu> python -m triton_kernels.xxfp4)
# =========================================================================== #
def _unit_test(
        device: torch.device = torch.device('cuda'),
) -> None:
    """
    Unit test vs the PyTorch reference quantize_fp4.rtn_xxfp4 (same algorithm: AbsMax group scale + RNE
    e2m1). Both MXFP4 (e8m0) and NVFP4 (e4m3) quantize bit-exactly vs the reference -- the e2m1 codes and
    the stored scale match exactly -- because the kernel rounds the scale to nearest (to_e8m0 / to_e4m3)
    and divides with div_rn (IEEE). Also checks self-consistency (pack -> dequantize == fake-quant).
    """

    from quantize_fp4 import rtn_xxfp4, dequant_xxfp4

    torch.manual_seed(seed=0)
    # MXFP4 (e8m0) and NVFP4 (e4m3) quantize BIT-EXACTLY vs the PyTorch reference at every output dtype:
    # to_e8m0 / to_e4m3 round the scale to nearest (matching torch) and the kernel divides with div_rn
    # (IEEE), so the e2m1 codes, the stored scale, AND the fp32/16-bit fake-quant all match exactly. The
    # dequant is grid * (scale_q * gs) -- (scale_q * gs) is a correctly-rounded multiply, so the dequant is a
    # single correctly-rounded op matching the reference.
    presets: dict = {  # the xxfp4 fields per format (see the module docstring)
        'mxfp4': {'group_size': 32, 'scale_dtype': torch.float8_e8m0fnu, 'scale_scale': 4., 'global_scale': 1. / 3.},
        'nvfp4': {'group_size': 16, 'scale_dtype': torch.float8_e4m3fn, 'scale_scale': 6., 'global_scale': .1},
    }
    for name, cfg in presets.items():
        for dtype in torch.bfloat16, torch.float16, torch.float32:
            x: torch.Tensor = torch.randn(512, 4096, dtype=dtype, device=device) * 2.5  # (M, K)
            fake_quant: torch.Tensor = fake_quantize(x=x, **cfg)  # (M, K)
            packed, scale = quantize_pack(x=x, **cfg)  # (M, K//2) float4_e2m1fn_x2, (M, K//G) scale_dtype
            dequant: torch.Tensor = dequantize(packed=packed, scale=scale, global_scale=cfg['global_scale'], dtype=dtype)  # (M, K)
            reference: dict = rtn_xxfp4(x=x, fp4_rounding_mode='even', high_dtype=torch.float32, round_dtype=dtype, **cfg)
            assert dequant.equal(fake_quant)  # pack -> dequantize round-trips the fake-quant (self-consistent, exact)
            assert scale.view(dtype=torch.uint8).equal(reference['scale_quant'].view(dtype=torch.uint8))  # group scale bit-exact
            assert packed.view(dtype=torch.uint8).equal(reference['e2m1'].view(dtype=torch.uint8))  # e2m1 codes byte-exact (the +-0 sign bit matches the cvt)
            assert scale.dtype == cfg['scale_dtype'] and packed.dtype == torch.float4_e2m1fn_x2 and fake_quant.dtype == dtype
            assert fake_quant.equal(reference['fake_quant']), (name, dtype, 'fake_quant')

        # corner-case groups, per output dtype: all-zero / all-inf / all-NaN match the reference.
        # The value-level fake-quant compare (both-NaN == equal) ALLOWS both sides NaN but with different mantissa payloads.
        for dt in torch.bfloat16, torch.float64:
            hd: torch.dtype = torch.float64 if dt == torch.float64 else torch.float32
            for fill in 0., torch.inf, torch.nan:
                xc: torch.Tensor = torch.full((cfg['group_size'],), fill, dtype=dt, device=device)  # one corner group, 1D
                fq_c: torch.Tensor = fake_quantize(x=xc, **cfg)  # (G,)
                pk_c, sc_c = quantize_pack(x=xc, **cfg)  # (G//2,) float4_e2m1fn_x2, (1,) scale
                ref_c: dict = rtn_xxfp4(x=xc, fp4_rounding_mode='even', high_dtype=hd, round_dtype=dt, **cfg)
                assert sc_c.view(dtype=torch.uint8).equal(ref_c['scale_quant'].view(dtype=torch.uint8)), (name, dt, fill, 'scale')  # scale byte bit-exact
                assert pk_c.view(dtype=torch.uint8).equal(ref_c['e2m1'].view(dtype=torch.uint8)), (name, dt, fill, 'codes')  # codes byte-exact
                assert ((fq_c == ref_c['fake_quant']) | (fq_c.isnan() & ref_c['fake_quant'].isnan())).all(), (name, dt, fill, 'fake_quant')  # value-equal; both-NaN == equal (allows the fp64 NaN-payload diff = A2)

        # existing_scale (HAS_SCALE path): round x against a SUPPLIED quantized scale -- the kernel skips amax,
        # loads + decodes the given scale (for nvfp4 on sm8x this is the int8 existing-scale load + from_e4m3
        # decode), rounds x against it, and returns it unchanged. Bit-exact vs the reference fed the same scale.
        xa: torch.Tensor = torch.randn(512, 4096, dtype=torch.bfloat16, device=device) * 2.5
        xb: torch.Tensor = torch.randn(512, 4096, dtype=torch.bfloat16, device=device) * 2.5  # different data -> a genuinely external scale
        _, es = quantize_pack(x=xa, **cfg)  # a valid quantized scale (scale_dtype) computed from xa
        es_fq: torch.Tensor = fake_quantize(x=xb, existing_scale=es.clone(), **cfg)
        es_pk, es_sc = quantize_pack(x=xb, existing_scale=es.clone(), **cfg)
        es_ref: dict = rtn_xxfp4(x=xb, existing_scale=es, fp4_rounding_mode='even', high_dtype=torch.float32, round_dtype=torch.bfloat16, **cfg)
        assert es_sc.view(dtype=torch.uint8).equal(es.view(dtype=torch.uint8)), (name, 'existing_scale returned unchanged')  # scale passes through verbatim
        assert es_pk.view(dtype=torch.uint8).equal(es_ref['e2m1'].view(dtype=torch.uint8)), (name, 'existing_scale codes')  # codes byte-exact
        assert es_fq.equal(es_ref['fake_quant']), (name, 'existing_scale fake_quant')
        assert dequantize(packed=es_pk, scale=es_sc, global_scale=cfg['global_scale'], dtype=torch.bfloat16).equal(es_fq), (name, 'existing_scale roundtrip')

        # inference_compat + existing_scale (e8m0 only): the supplied scale is in the stored f-2 convention, so
        # quantize shifts it +2 back to the kernel's default before rounding (line 346) and returns it unchanged.
        # Same codes / fake-quant as the default-convention existing_scale above; dequantize(+compat) restores.
        if cfg['scale_dtype'] == torch.float8_e8m0fnu:
            es_compat: torch.Tensor = compat_shift(es, -2)  # f-2 stored representation of the same (default-convention) es
            ic_fq: torch.Tensor = fake_quantize(x=xb, existing_scale=es_compat.clone(), inference_compat=True, **cfg)
            ic_pk, ic_sc = quantize_pack(x=xb, existing_scale=es_compat.clone(), inference_compat=True, **cfg)
            assert ic_sc.view(dtype=torch.uint8).equal(es_compat.view(dtype=torch.uint8)), (name, 'inference_compat existing_scale returned unchanged')  # the f-2 byte passes through verbatim
            assert ic_pk.view(dtype=torch.uint8).equal(es_ref['e2m1'].view(dtype=torch.uint8)), (name, 'inference_compat existing_scale codes')  # same codes as the default-convention scale
            assert ic_fq.equal(es_ref['fake_quant']), (name, 'inference_compat existing_scale fake_quant')
            assert dequantize(packed=ic_pk, scale=ic_sc, global_scale=cfg['global_scale'], dtype=torch.bfloat16, inference_compat=True).equal(ic_fq), (name, 'inference_compat existing_scale roundtrip')

    # FTZ guard: these two outcomes hold ONLY because the launches pass enable_reflect_ftz=False (IEEE
    # subnormals). If a future Triton re-enables flush-to-zero, libdevice would flush and these fail loudly:
    # (1) a subnormal e8m0 scale would collapse to byte 0 (not 1); (2) a -inf in a saturating group would
    # become +inf (inf * flushed-subnormal = NaN -> +grid-max). Both are mxfp4 (e8m0) only.
    cfg: dict = {'group_size': 32, 'scale_dtype': torch.float8_e8m0fnu, 'scale_scale': 4., 'global_scale': 1. / 3.}
    tiny: torch.Tensor = torch.full((32,), torch.finfo(torch.bfloat16).tiny, dtype=torch.bfloat16, device=device)  # amax/(ss*gs) is an fp32 subnormal
    assert quantize_pack(x=tiny, **cfg)[1].view(dtype=torch.uint8)[0].item() == 1, 'FTZ regression: subnormal e8m0 scale flushed to byte 0'
    sat: torch.Tensor = torch.full((32,), 0.5, dtype=torch.bfloat16, device=device)
    sat[0] = -torch.inf  # scale saturates to e8m0 byte 254
    assert fake_quantize(x=sat, **cfg)[0].item() == -torch.inf, 'FTZ regression: -inf in a saturating group flipped to +inf'

    # FTZ guard for the DEQUANT kernel: a stored byte-0 e8m0 scale decodes to 2^-127 (an fp32 subnormal), so
    # nonzero codes dequantize to the subnormal code * (2^-127 * gs); this must match the reference and stay
    # nonzero -- enable_reflect_ftz=False keeps it (FTZ would flush scale_q*gs to 0). e8m0 only.
    sc0: torch.Tensor = torch.zeros(2, dtype=torch.float8_e8m0fnu, device=device)  # byte 0 -> 2^-127
    pk0: torch.Tensor = torch.full((32,), 0x33, dtype=torch.uint8, device=device).view(dtype=torch.float4_e2m1fn_x2)  # nonzero codes (e2m1 grid 1.5), 2 groups of 32
    for dt0 in torch.bfloat16, torch.float32, torch.float64:
        hd0: torch.dtype = torch.float64 if dt0 == torch.float64 else torch.float32
        deq0: torch.Tensor = dequantize(packed=pk0, scale=sc0, global_scale=cfg['global_scale'], dtype=dt0)
        ref0: torch.Tensor = dequant_xxfp4(e2m1=pk0, scale_quant=sc0, global_scale=cfg['global_scale'], dtype=dt0, high_dtype=hd0, round_dtype=dt0)
        assert deq0.equal(ref0), ('dequant FTZ guard: byte-0 scale roundtrip', dt0)
        assert (deq0 != 0.).any(), ('FTZ regression: byte-0 e8m0 scale + nonzero codes dequantized to all-zero (subnormal flushed)', dt0)

    x: torch.Tensor = torch.randn(512, 4096, dtype=torch.bfloat16, device=device) * 2.5  # combined (all three outputs) + in-place
    fake_quant, packed, scale = quantize(x=x, do_dequant=True, do_pack=True)
    packed_only, scale_only = quantize_pack(x=x)
    assert fake_quant.equal(fake_quantize(x=x)) and packed.view(dtype=torch.uint8).equal(packed_only.view(dtype=torch.uint8)) and scale.view(dtype=torch.uint8).equal(scale_only.view(dtype=torch.uint8))
    x_inplace: torch.Tensor = x.clone()
    returned: torch.Tensor = fake_quantize(x=x_inplace, inplace=True)
    assert returned.data_ptr() == x_inplace.data_ptr() and x_inplace.equal(fake_quantize(x=x))

    # inference_compat (e8m0 only): the stored exponent byte shifts by -2, dequantize restores it -> identical fake-quant
    packed_c, scale_c = quantize_pack(x=x, inference_compat=True)
    assert dequantize(packed=packed_c, scale=scale_c, inference_compat=True).equal(fake_quantize(x=x))

    print('Unit test passed.')


def _benchmark(
        device: torch.device = torch.device('cuda'),
) -> None:
    """
    Throughput (input GB/s) of the MXFP4 quant kernels vs the PyTorch reference rtn_xxfp4.
    """

    from quantize_fp4 import rtn_xxfp4

    cfg: dict = {'group_size': 32, 'scale_dtype': torch.float8_e8m0fnu, 'scale_scale': 4., 'global_scale': 1. / 3.}
    print(f"{'K':>8} | {'fake_quant':>11} | {'quant_pack':>11} | {'rtn_xxfp4':>11}    (input GB/s, mxfp4)")
    for K in (2 ** i for i in range(10, 16)):
        x: torch.Tensor = torch.randn(4096, K, dtype=torch.bfloat16, device=device)
        gbps = lambda fn: x.numel() * x.element_size() * 1e-9 / (triton.testing.do_bench(fn) * 1e-3)  # bytes of x read, GB/s
        print(f'{K:>8} | {gbps(lambda: fake_quantize(x=x, **cfg)):>11.1f} | {gbps(lambda: quantize_pack(x=x, **cfg)):>11.1f} | {gbps(lambda: rtn_xxfp4(x=x, fp4_rounding_mode="even", **cfg)):>11.1f}')


if __name__ == '__main__':
    _unit_test(device=torch.device('cuda'))
    _benchmark(device=torch.device('cuda'))
