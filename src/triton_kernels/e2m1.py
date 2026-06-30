"""
E2M1 (FP4) format primitives + plain BF16 <-> E2M1 conversion (NO group scales).

This module owns the low-level e2m1 building blocks that everything else reuses
(xxfp4.py, fused.py): rounding to a code, code<->value, and nibble pack/unpack.
The "no group scales" path here rounds each value directly onto the e2m1 grid
{0,+-0.5,+-1,+-1.5,+-2,+-3,+-4,+-6} (clamping |x|<=6) -- useful for pre-normalized data
and as the format-conversion building block.

Code layout: 4 bits = sign(bit 3) | magnitude(bits 0..2). magnitude grid index
0..7 -> {0,0.5,1,1.5,2,3,4,6}. Packed: 2 codes / uint8, low nibble = even element.
Rounding: round-to-nearest-EVEN via libdevice.rint on the piecewise-uniform grid.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.libdevice import rint

from triton_kernels.guard import device_guard, has_capability


# =========================================================================== #
# Composable device functions (call from any @triton.jit kernel)
# =========================================================================== #
@triton.jit
def round_to_code(v):
    """
    fp32 value -> int32 e2m1 code (sign|magnitude), RNE, clamps |v|<=6. Matches the Blackwell
    cvt.rn.satfinite.e2m1x2.f32: the sign BIT is copied for finite values (so -0.0 -> -0, +0.0 -> +0); NaN
    canonicalizes to +6 (sign dropped).
    """
    a = v.abs()
    mag = tl.where(
        a <= 2., rint(a * 2.),  # grid {0,.5,1,1.5,2} -> code 0..4
        tl.where(
            a <= 4., rint(a) + 2.,  # grid {2,3,4} -> code 4..6
            tl.where(a <= 5., 6., 7.)
        )
    ).to(tl.int32)  # grid {4,6}, RNE tie@5 + clip -> code 6/7 (>5 / inf / NaN -> 7)
    # sign = v < 0, EXCEPT a zero magnitude takes the input's SIGN BIT so a literal -0.0 -> -0 (like the cvt).
    # `v < 0` is False for -0.0; in the zero-mag region (small finite v, any float dtype) `1/v` carries the sign
    # (-0.0 -> -inf < 0). NaN/inf have mag != 0, so they keep the `v < 0` branch (NaN -> +6, -inf -> -6).
    neg = tl.where(mag == 0, (1. / v) < 0, v < 0).to(tl.int32)
    return mag | (neg << 3)


@triton.jit
def code_to_value(code):
    """
    int32 e2m1 code -> fp32 grid value.
    """
    mag_code = code & 0x7
    neg = (code >> 3) & 1
    fm = mag_code.to(tl.float32)
    mag = tl.where(mag_code <= 4, fm * .5, tl.where(mag_code <= 6, fm - 2., 6.))
    return mag * tl.where(neg == 1, -1., 1.)  # signed-zero-safe (IEEE +0*-1 = -0): the -0 code (0x8) decodes to -0.0, bit-matching the hw cvt decode / torch grid (-mag would give +0 -- triton lowers it as 0-0)


@triton.jit
def round_to_value(v):
    """
    fp32 value -> e2m1 grid VALUE directly (RNE, clamp |v|<=6). Equals
    code_to_value(round_to_code(v)) but skips the integer code -- for the fake-quant path.
    """
    a = v.abs()
    mag = tl.where(
        a <= 2., rint(a * 2.) * .5,  # grid {0,.5,1,1.5,2}
        tl.where(
            a <= 4., rint(a),  # grid {2,3,4}
            tl.where(a <= 5., 4., 6.)
        )
    )  # grid {4,6}, RNE tie@5 + clip (>5 / inf / NaN -> 6)
    # sign mirrors round_to_code: a zero-magnitude result takes the input's SIGN BIT (1/v: -0.0 -> -inf < 0), so
    # -0.0 / small-negative-rounds-to-zero -> -0.0; NaN/inf keep `v < 0` (mag != 0). `mag * +-1` is signed-zero-safe.
    neg = tl.where(mag == 0., (1. / v) < 0, v < 0)
    return mag * tl.where(neg, -1., 1.)  # fake-quant VALUE, bit-matching the packed code's sign (and the hw cvt / torch grid)


# pack / unpack are exact inverses (tl.split / tl.join). The caller reshapes to/from the
# trailing pair dim -- packing is a 1D op on consecutive pairs, so no length argument is needed.
@triton.jit
def pack(pair):
    """
    pair: int codes shaped (..., 2) = [low nibble, high nibble] -> packed uint8 (...).
    """
    lo, hi = pair.split()
    return ((lo & 0xF) | ((hi & 0xF) << 4)).to(tl.uint8)


@triton.jit
def unpack(packed):
    """
    packed: uint8 (...) -> int32 codes (..., 2) = [low nibble, high nibble].
    """
    p = packed.to(tl.int32)
    return tl.join(p & 0xF, (p >> 4) & 0xF)


# =========================================================================== #
# Direct hardware e2m1 conversion (Blackwell sm_100+) vs the portable libdevice path, selected by HW_CVT.
# These three helpers are the ONLY place the cvt.*.e2m1x2 instructions appear in the package -- xxfp4.py /
# fused.py import and call them (cross-module @triton.jit calls, like code_to_value / pack already are). They
# take / return WHOLE tiles (last axis = the elements to convert); the pair split/join is the cvt's packing
# detail, so it lives inside the HW_CVT branch and the libdevice path stays whole-tile (round_to_value /
# round_to_code straight on the tile). HW_CVT is a tl.constexpr: when True the inline-asm cvt is emitted; when
# False the whole branch -- and thus the cvt PTX, which only assembles on sm_100+/sm_120+ -- is compiled out,
# keeping the kernels portable. The hw and libdevice paths are bit-identical (round_to_code was written to
# match cvt.rn.satfinite.e2m1x2.f32; the special values inf/NaN/-0.0 and the nibble order are pinned by
# _unit_test). Constraint/operand details were verified against ptxas + the reference on sm_100.
# =========================================================================== #
@triton.jit
def round_pack(x, HW_CVT: tl.constexpr):
    """
    fp32 tile (..., G) -> packed uint8 e2m1 bytes (..., G//2): consecutive pairs (even -> low nibble, odd ->
    high nibble). HW_CVT: split the pair axis and emit one Blackwell cvt.rn.satfinite.e2m1x2.f32 per pair
    (operand order $0,$2,$1 -> lo source $1 -> LOW nibble, hi $2 -> HIGH nibble, matching qutlass
    fp32_vec_to_e2m1 and the pack() convention); the e2m1x2 byte is produced in a .b8 then zero-extended to the
    uint8 output. Else round_to_code the whole tile + pack -- byte-identical.
    """
    # NB: inline the pair shape in each branch -- a shape tuple bound to a local that crosses the HW_CVT
    # constexpr branch gets promoted to a tensor (reshape then rejects it).
    if HW_CVT:
        lo, hi = x.reshape(x.shape[:-1] + (x.shape[-1] // 2, 2)).split()
        return tl.inline_asm_elementwise(
            '{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; cvt.u32.u8 $0, t; }',
            '=r,f,f',
            [lo, hi],
            dtype=tl.uint8, is_pure=True, pack=1,
        )
    return pack(round_to_code(x).reshape(x.shape[:-1] + (x.shape[-1] // 2, 2)))


@triton.jit
def dequant(byte, HW_CVT: tl.constexpr):
    """
    packed uint8 e2m1 bytes (..., G//2) -> fp32 grid values (..., G) (low nibble = even, high = odd). HW_CVT:
    one cvt.rn.f16x2.e2m1x2 per byte (moved into a .b8 source, decoded to an f16x2 = low/high nibble, upcast to
    fp32 -- every e2m1 grid value is exact in f16, so lossless and bit-identical to code_to_value, signed zero
    included) then join. Else unpack + code_to_value. Whole-tile either way.
    """
    # NB: inline the flattened shape in each branch (see round_pack) -- a local shape tuple that crosses the
    # HW_CVT constexpr branch gets promoted to a tensor, which reshape rejects.
    if HW_CVT:
        lo, hi = tl.inline_asm_elementwise(
            '{ .reg .b8 b0, b1, b2, b3; .reg .b32 p; mov.b32 {b0, b1, b2, b3}, $2; cvt.rn.f16x2.e2m1x2 p, b0; mov.b32 {$0, $1}, p; }',
            '=h,=h,r',
            [byte],
            dtype=(tl.float16, tl.float16), is_pure=True, pack=1,
        )
        return tl.join(lo, hi).reshape(byte.shape[:-1] + (byte.shape[-1] * 2,)).to(tl.float32)
    return code_to_value(unpack(byte)).reshape(byte.shape[:-1] + (byte.shape[-1] * 2,))


@triton.jit
def round_value(x, HW_CVT: tl.constexpr):
    """
    fp32 tile (..., G) -> e2m1 grid values (..., G), WITHOUT producing the packed bytes -- the fake-quant
    (dequant-only) path. HW_CVT has no non-byte route, so it round-trips through the cvt bytes (round_pack ->
    dequant); the libdevice path rounds straight to the value (round_to_value, which skips the integer code),
    avoiding the pack/unpack a byte detour would cost. Bit-identical either way.
    """
    if HW_CVT:
        return dequant(round_pack(x, True), True)
    return round_to_value(x)


# =========================================================================== #
# Standalone elementwise kernels (no scale)
# =========================================================================== #
# Small, portable autotune set for the pure-streaming elementwise kernels. BLOCK is the number
# of ELEMENTS processed per program (must be even). Shared with xxfp4.py, which derives its
# groups-per-program as BLOCK // G. Throughput is robust to these; a few cover other GPUs.
ELT_CONFIGS = [
    triton.Config({'BLOCK': 2048},  num_warps=4),
    triton.Config({'BLOCK': 4096},  num_warps=8),
    triton.Config({'BLOCK': 8192},  num_warps=8),
    triton.Config({'BLOCK': 16384}, num_warps=8),
    # sm_100 (B200, HBM3e ~5.6 TB/s): the per-element body (round / pack ALU) is exposed by the fast HBM, so
    # SMALLER blocks + fewer warps (less per-thread work, no register spill) win here; the big-BLOCK configs
    # above spill and collapse to <2000 GB/s. Slower-memory GPUs keep selecting the entries above.
    triton.Config({'BLOCK': 1024},  num_warps=2),
    triton.Config({'BLOCK': 2048},  num_warps=2),
    triton.Config({'BLOCK': 2048},  num_warps=8),
    triton.Config({'BLOCK': 4096},  num_warps=4),
    triton.Config({'BLOCK': 8192},  num_warps=2),
]


# e2m1 quant is elementwise (no group reduction), so the kernel is fully 1D in BLOCK elements.
# DO_DEQUANT and DO_PACK are independent: emit the fake-quant tensor, the packed codes, or both from a
# single pass. When packing, the e2m1 BYTE is computed once (round_pack) and reused -- packed straight out, and
# decoded back (dequant) for the fake-quant store, so the value is always consistent with the byte. The
# fake-quant-only path rounds straight to the grid value (round_value). DO_DEQUANT may be inplace (y_ptr == x_ptr).
@triton.autotune(configs=ELT_CONFIGS, key=['DO_DEQUANT', 'DO_PACK', 'HW_CVT'])
@triton.jit
def _quant_kernel(
        x_ptr, y_ptr, q_ptr,
        n_elements,
        DO_DEQUANT: tl.constexpr, DO_PACK: tl.constexpr, HW_CVT: tl.constexpr,
        BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    e = pid * BLOCK + tl.arange(0, BLOCK)  # element offsets
    x = tl.load(x_ptr + e, mask=e < n_elements).to(tl.float32)
    if DO_PACK:
        byte = round_pack(x, HW_CVT)  # uint8 (BLOCK // 2,) -- single source for both stores
        if DO_DEQUANT:
            tl.store(y_ptr + e, dequant(byte, HW_CVT), mask=e < n_elements)  # consistent with the packed byte
        b = pid * (BLOCK // 2) + tl.arange(0, BLOCK // 2)  # byte offsets (contiguous)
        tl.store(q_ptr + b, byte, mask=b < n_elements // 2)
    else:  # DO_DEQUANT only: round straight to the grid value (no byte)
        tl.store(y_ptr + e, round_value(x, HW_CVT), mask=e < n_elements)  # store auto-casts to y dtype


@triton.autotune(configs=ELT_CONFIGS, key=['HW_CVT'])
@triton.jit
def _dequant_kernel(
        q_ptr, y_ptr,
        n_elements,
        HW_CVT: tl.constexpr,
        BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid * (BLOCK // 2) + tl.arange(0, BLOCK // 2)
    e = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(y_ptr + e, dequant(tl.load(q_ptr + b, mask=b < n_elements // 2), HW_CVT), mask=e < n_elements)  # store auto-casts to y dtype


# =========================================================================== #
# Python API
# =========================================================================== #
def quantize(x: torch.Tensor, *, do_dequant: bool = True, do_pack: bool = False, inplace: bool = False):
    """
    E2M1 quantize (...,K), no scale, elementwise.
      dequant: produce the fake-quantized (round-tripped) tensor.
      pack:    produce packed e2m1 codes, float4_e2m1fn_x2 (...,K//2) (2 codes per byte).
      inplace: write the fake-quant result back into x (requires dequant and a contiguous x).
    Returns the fake-quant tensor (dequant only), the packed codes (pack only), or both as a tuple.
    """
    assert do_dequant or do_pack, 'select at least one of dequant / pack'
    assert not (inplace and not do_dequant), 'inplace requires dequant=True'
    assert x.numel() % 2 == 0
    device: torch.device = x.device
    if inplace:
        assert x.is_contiguous(), 'inplace requires a contiguous tensor'
        xc: torch.Tensor = x
    else:
        xc: torch.Tensor = x.contiguous()
    n: int = xc.numel()
    # outputs are allocated in their final shape; the kernel only uses flat (contiguous) offsets.
    y: torch.Tensor = xc if (inplace or not do_dequant) else torch.empty_like(xc)  # xc is a dummy when not dequant
    q: torch.Tensor = torch.empty(*x.shape[:-1], x.size(-1) // 2, dtype=torch.float4_e2m1fn_x2, device=device) if do_pack else torch.empty(0, dtype=torch.float4_e2m1fn_x2, device=device)
    hw_cvt: bool = has_capability(min_major=10, device_index=device.index)  # Blackwell sm_100+: emit the native e2m1 cvt
    with device_guard(device):
        _quant_kernel[lambda m: (triton.cdiv(n, m['BLOCK']),)](xc, y, q.view(dtype=torch.uint8), n_elements=n, DO_DEQUANT=do_dequant, DO_PACK=do_pack, HW_CVT=hw_cvt)
    if do_dequant and do_pack:
        return y, q
    return y if do_dequant else q


def fake_quantize(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """
    Round (...,K) to the E2M1 grid and back (no scale). -> same shape/dtype (in place if inplace=True).
    """
    return quantize(x=x, do_dequant=True, do_pack=False, inplace=inplace)


def quantize_pack(x: torch.Tensor) -> torch.Tensor:
    """
    Quantize + pack (...,K) -> packed float4_e2m1fn_x2 (...,K//2). No scales.
    """
    return quantize(x=x, do_dequant=False, do_pack=True)


def dequantize(packed: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """
    Unpack + dequantize packed float4_e2m1fn_x2 (...,K//2) -> (...,K). No scales.
    """
    pc: torch.Tensor = packed.contiguous()
    n: int = pc.numel() * 2
    y: torch.Tensor = torch.empty(*packed.shape[:-1], packed.size(-1) * 2, dtype=dtype, device=packed.device)   # final shape; flat offsets
    hw_cvt: bool = has_capability(min_major=10, device_index=packed.device.index)  # Blackwell sm_100+: emit the native e2m1 cvt
    with device_guard(packed.device):
        _dequant_kernel[lambda m: (triton.cdiv(n, m['BLOCK']),)](pc.view(torch.uint8), y, n_elements=n, HW_CVT=hw_cvt)  # view FP4 bytes as uint8 for the kernel
    return y


# =========================================================================== #
# Unit test + benchmark  (run from the source root:  CUDA_VISIBLE_DEVICES=<gpu> python -m triton_kernels.e2m1)
# =========================================================================== #
def _unit_test(
        device: torch.device = torch.device('cuda'),
) -> None:
    """
    Unit test: plain (no group scale) BF16/FP16/FP32/FP64 <-> E2M1 vs the PyTorch reference
    quantize_fp4.rtn_e2m1 / unpack_e2m1. The kernel rounds in fp32 (round-to-nearest-even) with no
    division, so it is bit-exact against the reference computed at the same precision. Also covers the
    special values +-inf (-> +-6) and NaN (-> +6), which saturate to the e2m1 grid max.
    """

    from quantize_fp4 import rtn_e2m1, unpack_e2m1

    torch.manual_seed(seed=0)
    for dtype in torch.bfloat16, torch.float16, torch.float32, torch.float64:
        for shape in (500, 4096), (4, 512, 2048):
            x: torch.Tensor = torch.randn(*shape, dtype=dtype, device=device) * 2.  # (..., C)
            ref_fake_quant, ref_packed = rtn_e2m1(x=x, mode='even', return_packed=True, high_dtype=torch.float32)  # (..., C), (..., C//2) float4_e2m1fn_x2
            fake_quant: torch.Tensor = fake_quantize(x=x)  # (..., C)
            packed: torch.Tensor = quantize_pack(x=x)  # (..., C//2) float4_e2m1fn_x2
            dequant: torch.Tensor = dequantize(packed=packed, dtype=dtype)  # (..., C)
            assert fake_quant.equal(ref_fake_quant.to(dtype=dtype))  # RNE onto the grid, no division -> bit-exact
            assert packed.view(dtype=torch.uint8).equal(ref_packed.view(dtype=torch.uint8))  # packed codes byte-exact vs the reference
            assert dequant.equal(fake_quant)  # unpack + dequant round-trips the fake-quant exactly
            assert unpack_e2m1(x_packed=packed, dtype=torch.float64).equal(fake_quant.to(dtype=torch.float64))  # packed is consistent with the fake-quant
            assert fake_quant.dtype == dtype and dequant.shape == x.shape and packed.shape == (*shape[:-1], shape[-1] // 2) and packed.dtype == torch.float4_e2m1fn_x2

    # +-inf and NaN: round_to_code saturates |.| > 5 -- and NaN, whose comparisons are all false -- to the
    # grid max, so +-inf -> +-6 and NaN -> +6 (sign bit ignored). Bit-exact vs the reference.
    for dtype in torch.bfloat16, torch.float16, torch.float32, torch.float64:
        special: torch.Tensor = torch.as_tensor([torch.inf, -torch.inf, torch.nan, -torch.nan], dtype=dtype, device=device)  # (4,)
        expect: torch.Tensor = torch.as_tensor([6., -6., 6., 6.], dtype=dtype, device=device)  # +-inf -> +-6; NaN -> +6
        sp_fake: torch.Tensor = fake_quantize(x=special)  # (4,)
        sp_dequant: torch.Tensor = dequantize(packed=quantize_pack(x=special), dtype=dtype)  # (4,) via pack -> unpack
        ref_special, _ = rtn_e2m1(x=special, mode='even', return_packed=True, high_dtype=torch.float32)  # (4,)
        assert sp_fake.equal(expect)  # +-inf -> +-6 (saturate to the grid max), NaN -> +6
        assert sp_fake.equal(ref_special.to(dtype=dtype))  # matches the PyTorch reference
        assert sp_dequant.equal(sp_fake)  # pack -> dequant round-trips the fake-quant

    # signed zero: the cvt copies the input's sign bit -- -0.0 -> -0 (0x8), +0.0 -> +0 (0x0), and small values
    # round to a SIGNED zero -- so the kernel and reference are byte-identical (verified vs cvt.rn.satfinite.e2m1x2.f32).
    for dtype in torch.bfloat16, torch.float16, torch.float32, torch.float64:
        z: torch.Tensor = torch.as_tensor([-0., 0., -0.2, 0.2, -0.1, 0.1, -0.25, 0.25], dtype=dtype, device=device)  # (8,)
        z_bytes: torch.Tensor = quantize_pack(x=z).view(dtype=torch.uint8)
        codes: list = torch.stack([z_bytes & 0xF, (z_bytes >> 4) & 0xF], dim=-1).flatten().tolist()  # low nibble = even elem
        assert codes == [0x8, 0x0, 0x8, 0x0, 0x8, 0x0, 0x8, 0x0], (dtype, 'signed-zero codes', codes)  # -0.0/-x -> -0 (0x8); +0.0/+x -> +0 (0x0), like the cvt
        _, z_ref = rtn_e2m1(x=z, mode='even', return_packed=True, high_dtype=torch.float32)
        assert z_bytes.equal(z_ref.view(dtype=torch.uint8)), (dtype, 'signed-zero kernel != reference')  # codes byte-exact vs the reference
        # DECODE must also carry the sign BIT: the -0 code (0x8) dequants to -0.0, bit-matching the hw cvt decode
        # (cvt.rn.f16x2.e2m1x2) and the torch grid (unpack_e2m1 index 8 == -0.); fake_quantize (round_to_value) too.
        expect_sign: torch.Tensor = torch.as_tensor([True, False, True, False, True, False, True, False], device=device)  # -0 / +0 per element
        assert dequantize(packed=quantize_pack(x=z), dtype=dtype).signbit().equal(expect_sign), (dtype, 'signed-zero DECODE signbit')  # code_to_value
        assert fake_quantize(x=z).signbit().equal(expect_sign), (dtype, 'signed-zero fake-quant signbit')  # round_to_value
        assert unpack_e2m1(x_packed=quantize_pack(x=z), dtype=torch.float64).signbit().equal(expect_sign), (dtype, 'signed-zero torch-ref signbit')  # torch grid agrees

    x: torch.Tensor = torch.randn(500, 4096, dtype=torch.bfloat16, device=device) * 2.  # combined (both outputs in one pass) + in-place
    fake_quant, packed = quantize(x=x, do_dequant=True, do_pack=True)
    assert fake_quant.equal(fake_quantize(x=x)) and packed.view(torch.uint8).equal(quantize_pack(x=x).view(torch.uint8))
    x_inplace: torch.Tensor = x.clone()
    returned: torch.Tensor = fake_quantize(x=x_inplace, inplace=True)
    assert returned.data_ptr() == x_inplace.data_ptr() and x_inplace.equal(fake_quantize(x=x))

    print('Unit test passed.')


def _benchmark(
        device: torch.device = torch.device('cuda'),
) -> None:
    """
    Throughput (input GB/s) of the no-scale E2M1 kernels vs the PyTorch reference rtn_e2m1.
    """

    from quantize_fp4 import rtn_e2m1

    print(f"{'K':>8} | {'fake_quant':>11} | {'quant_pack':>11} | {'rtn_e2m1':>11}    (input GB/s)")
    for K in (2 ** i for i in range(10, 16)):
        x: torch.Tensor = torch.randn(4096, K, dtype=torch.bfloat16, device=device)
        gbps = lambda fn: x.numel() * x.element_size() * 1e-9 / (triton.testing.do_bench(fn) * 1e-3)  # bytes of x read, GB/s
        print(f'{K:>8} | {gbps(lambda: fake_quantize(x=x)):>11.1f} | {gbps(lambda: quantize_pack(x=x)):>11.1f} | {gbps(lambda: rtn_e2m1(x=x, mode="even", return_packed=True)):>11.1f}')


if __name__ == '__main__':
    _unit_test(device=torch.device('cuda'))
    _benchmark(device=torch.device('cuda'))
