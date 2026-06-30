"""
Triton MXFP4 / NVFP4 matmul.

Computes C = alpha * (dequant(A) @ dequant(B).T), with A (M, K) and B (N, K). C is bf16 by default; pass
out_dtype= for fp16 or fp32. Both operands are stored K-contiguous and we form A @ B.T -- the layout the FP4
tensor cores want (the "tn" in the CUTLASS kernel names).

DATA LAYOUT (both operands):
  * 4-bit E2M1 codes packed two-per-byte, low nibble = lower K index:
        byte bits:  7 6 5 4 | 3 2 1 0
                    [ code1 ] [ code0 ]      -> K elements 2j+1 and 2j
    so A is uint8 (M, K//2) and B is uint8 (N, K//2).
  * one block scale per group of K elements, in one of two hardware formats:
        MXFP4: group = 32, scale = E8M0 (power-of-two, a uint8 exponent)
        NVFP4: group = 16, scale = E4M3 (float8_e4m3fn)
    Only these two are supported: tl.dot_scaled derives the group size from the scale dtype (E8M0/uint8 ->
    32, E4M3 -> 16), so the (group, dtype) pairing is fixed even though the hardware allows mixed.

COMPUTE PATHS. The two tl.dot_scaled paths emit the native FP4 tensor-core MMA on sm_120 / RTX 5090
(confirmed in the PTX: mma.sync...kind::mxf4nvf4.block_scale...); they differ only in how operands + scales
are loaded. A third path, dequant -> cuBLAS, is the pre-Blackwell fallback (see "PRE-BLACKWELL" below):
  * pointer loads -> fp4_matmul(...): tl.dot_scaled, tl.load with plain row-major scales; best in the
                     memory-bound DECODE regime (small M). (NVFP4 pre-Blackwell falls back to dequant here.)
  * TMA loads     -> fp4_matmul_tma(...): tl.dot_scaled, streaming operands and swizzled scales via Tensor
                     Memory Accelerator descriptors; best in the compute-bound PREFILL regime on Blackwell.
                     (After Triton's 10-block-scaled-matmul.py.)
  * dequant       -> _fp4_matmul_dequant(...): dequantize both operands to out_dtype (xxfp4.dequantize) +
                     one cuBLAS GEMM; the PREFILL path on pre-Blackwell (TMA is dead there) for both formats.
  * auto-dispatch -> fp4_matmul_auto(...): takes row-major scales. On Blackwell routes small-M to the pointer
                     path and large-M to TMA (swizzling scales on the fly + zero-padding K to a multiple of 128
                     via pad_k_to_tile, falling back to pointer only when N is not (16//itemsize)-aligned). On
                     pre-Blackwell routes large-M to dequant and small-M to the pointer path.
Each entry point infers the format (mxfp4 / nvfp4) from the scale dtype -- E8M0 (float8_e8m0fnu) vs E4M3
(float8_e4m3fn); pass out_dtype=bf16/fp16/fp32.

PRE-BLACKWELL NVFP4 (e4m3 scales): tl.dot_scaled's upcast fallback only supports e8m0 (uint8) scales, so an
e4m3 scale crashes the compiler on sm_89 / sm_86 / Hopper (TritonGPUAccelerateMatmul isIntOrIndex assertion).
MXFP4 (e8m0) works everywhere via that fallback. For NVFP4 below Blackwell (fp4_blockscale_native False) we
instead dequantize both operands to out_dtype (xxfp4.dequantize) and do a single cuBLAS matmul --
_fp4_matmul_dequant. That beats an in-kernel manual dequant at every M on the L40S, and beats dot_scaled for
MXFP4 prefill too, so fp4_matmul_auto also routes MXFP4 large-M through it. NVFP4 via TMA stays Blackwell-only.

CUDA / HARDWARE CONCEPTS used below (everything else is plain Triton):
  * mma.sync       the hardware tensor-core matrix-multiply-accumulate instruction; "block_scale" variants
                   take per-group scales.
  * L2 reuse       program ids are ordered so neighboring programs share A/B tiles in the L2 cache (see
                   section 1, _grouped_pid).
  * shared memory  small on-chip scratch per program block; sm_120 has only ~100KB, which bounds the TMA
                   tile sizes (section 3).
  * num_warps      threads-per-program / 32 (a "warp" = 32 lanes in lockstep).
  * num_stages     software-pipeline depth: how many K-tiles are prefetched ahead of the MMA to hide memory
                   latency.
  * TMA descriptor a hardware async bulk-copy engine; see section 3.
  * block swizzle  the cuBLAS/CUTLASS "to_blocked" scale byte layout the tensor cores read; ASCII diagram in
                   section 3.

FILE STRUCTURE:
  1. shared helpers                _grouped_pid (+ L2 diagram)
  2. matmul, pointer loads         _MATMUL_CONFIGS, _fp4_matmul_kernel, _fp4_matmul_dequant, fp4_matmul
  3. matmul, TMA loads             _MATMUL_TMA_CONFIGS, _fp4_matmul_tma_kernel, fp4_matmul_tma (+ swizzle diagram)
  4. scale swizzle + auto-router   _scale_swizzle_kernel, to_blocked / to_blocked_torch, pad_k_to_tile, fp4_matmul_auto
  5. correctness + benchmark       _unit_test, _benchmark

Run: python -m triton_kernels.fp4mm  (runs the unit test then the benchmark).
"""

import functools
import itertools

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from triton_kernels.guard import device_guard, has_capability
from triton_kernels.xxfp4 import dequantize  # FP4 -> bf16/fp16/fp32, for the pre-Blackwell dequant -> cuBLAS path


@functools.lru_cache(maxsize=None)
def fp4_blockscale_native(device_index: int) -> bool:
    """
    Does this device have the native FP4 block-scale tensor-core MMA (mma.sync...mxf4nvf4.block_scale) that
    tl.dot_scaled needs for NVFP4 (e4m3) scales? Blackwell only (sm_100 / sm_120, cc >= 10). On pre-Blackwell
    (incl. Hopper cc 9 and Ada/Ampere cc 8) tl.dot_scaled has only the upcast fallback, which supports e8m0
    (uint8) scales but NOT e4m3 -- an e4m3 scale crashes the TritonGPUAccelerateMatmul pass (isIntOrIndex
    assertion). So NVFP4 there goes through _fp4_matmul_dequant (dequant -> cuBLAS); MXFP4 (e8m0) uses
    tl.dot_scaled everywhere. The TMA path also needs this (it feeds tl.dot_scaled).
    """
    return has_capability(min_major=10, device_index=device_index)


# -----------------------------------------------------------------------------
# 1. Shared helpers (used by both compute paths): _grouped_pid
# -----------------------------------------------------------------------------
# _grouped_pid maps the 1-D program id to an output tile, ordered for L2 cache reuse.
# C is tiled into BLOCK_M x BLOCK_N output blocks; each Triton program computes
# one block, reading a full strip of A (its BLOCK_M rows, all K) and a strip of B
# (its BLOCK_N rows, all K). The order in which the 1-D program id maps to blocks
# decides which A/B strips sit in L2 together.
#
# "Grouping" splits the row-tiles into horizontal BANDS of GROUP_M rows. Program
# ids fill one band completely -- walking DOWN each column, then moving right --
# before the next band. The payoff: the GROUP_M programs that share a B strip run
# back-to-back, so that strip stays hot in L2 instead of being re-fetched from
# DRAM on every step (which is what naive row-major order does).
#
# Example: 4 x 4 grid of output tiles, cell = the program id that computes it.
#
#   naive row-major  (id steps along a row -> a different B strip every step):
#         n0  n1  n2  n3
#    m0    0   1   2   3
#    m1    4   5   6   7
#    m2    8   9  10  11
#    m3   12  13  14  15
#
#   grouped, GROUP_M = 2  (id fills a 2-row band column-by-column, then steps
#   right; the brackets at right mark the two bands):
#         n0  n1  n2  n3
#    m0    0   2   4   6   <-.
#    m1    1   3   5   7   <-'  band 0 = ids 0..7   (rows m0, m1)
#    m2    8  10  12  14   <-.
#    m3    9  11  13  15   <-'  band 1 = ids 8..15  (rows m2, m3)
#
#   Reading band 0 in id order 0,1,2,3,4,5,6,7 visits B strips n0,n0,n1,n1,n2,
#   n2,n3,n3: each strip is used by GROUP_M=2 consecutive ids (the 2nd is an L2
#   hit), so every B strip is fetched from DRAM once per band, not once per id.
#
# Both compute paths (sections 2 and 3) call this same _grouped_pid mapping.
@triton.jit
def _grouped_pid(M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_M: tl.constexpr):
    """
    Map the 1-D program id to (pid_m, pid_n) in the grouped order drawn above.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


# -----------------------------------------------------------------------------
# 2. matmul: pointer loads + tl.dot_scaled  (memory-bound / DECODE)
# -----------------------------------------------------------------------------
# One program computes C[BLOCK_M, BLOCK_N] = A_tile @ B_tile.T, looping over K in
# BLOCK_K chunks. The packed E2M1 codes and the row-major block scales feed
# straight into tl.dot_scaled, which lowers to the FP4 tensor-core MMA. dot_scaled
# infers the format from the scale dtype: E8M0/uint8 -> group 32 (MXFP4),
# E4M3/float8e4nv -> group 16 (NVFP4); the data format string stays 'e2m1'.
# Tiles span decode (small M, memory bound) and prefill (large M, compute bound); the autotuner benchmarks
# them per (M, N, K). GROUP_M=8 is the L2-reuse band height fed to _grouped_pid (section 1); the TMA path
# autotunes its own GROUP_M (see _MATMUL_TMA_CONFIGS).
_MATMUL_CONFIGS = [
    triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 256, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 256, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 256, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 256, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 256, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 256, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 256, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=4),
]


@functools.lru_cache(maxsize=None)
def _nvfp4_min_block_m(device_index: int) -> int:
    """
    Minimum BLOCK_M for the NVFP4 (e4m3-scale) pointer dot_scaled on this device. On sm_100 (B200 / GB200, cc
    (10, 0)) the native block-scaled FP4 MMA is a 128-row atom, and Triton's TritonGPUAccelerateMatmul
    miscompiles an e4m3-scaled dot_scaled for BLOCK_M < 128 -- the SAME isIntOrIndex assertion the pre-Blackwell
    upcast fallback hits (see fp4_blockscale_native) -- so force BLOCK_M >= 128 there. MXFP4 (e8m0/uint8 scales)
    is unaffected at any BLOCK_M, as is sm_120 (consumer Blackwell, which tolerates BLOCK_M < 128 here).
    Returns 0 (no constraint) off sm_100.
    """
    return 128 if has_capability(min_major=10, max_major=10, max_minor=0, device_index=device_index) else 0  # exactly (10, 0)


def _prune_fp4_matmul_configs(configs, named_args: dict = {}, **kwargs):
    """
    early_config_prune for _fp4_matmul_kernel: drop the BLOCK_M < 128 configs for the NVFP4 (GROUP_SIZE == 16,
    e4m3) pointer path on sm_100, where they crash the compiler (see _nvfp4_min_block_m). No-op for MXFP4
    (GROUP_SIZE == 32) and off sm_100. Always returns a non-empty list.
    """
    group_size = kwargs.get('GROUP_SIZE', named_args.get('GROUP_SIZE'))
    if group_size != 16:
        return list(configs)
    min_bm: int = _nvfp4_min_block_m(torch.cuda.current_device())  # set by the device_guard around the launch
    if not min_bm:
        return list(configs)
    kept = [c for c in configs if c.kwargs.get('BLOCK_M', 0) >= min_bm]
    return kept or list(configs)


# M is deliberately NOT in the autotune key: this decode path is tuned once per (N, K, GROUP_SIZE) instead of
# re-tuned for every batch size (cf. transform.py/fused.py). Verified perf-neutral -- the best(direct, tma)
# geomean is unchanged, since prefill (large M) uses fp4_matmul_tma anyway; only the direct path at large M
# (where you'd use the TMA path) is left on the decode-tuned config.
@triton.autotune(
    configs=_MATMUL_CONFIGS,
    key=['N', 'K', 'GROUP_SIZE'],
    prune_configs_by={'early_config_prune': _prune_fp4_matmul_configs},
)
@triton.jit
def _fp4_matmul_kernel(
        a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, c_ptr, alpha_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_asm, stride_ask,
        stride_bsn, stride_bsk,
        stride_cm, stride_cn,
        GROUP_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
):
    """
    C = alpha * (A @ B.T) via tl.dot_scaled. GROUP_SIZE = 32 MXFP4 / 16 NVFP4 (inferred from the scale dtype).
    """
    pid_m, pid_n = _grouped_pid(M, N, BLOCK_M, BLOCK_N, GROUP_M)
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    PACK_K: tl.constexpr = BLOCK_K // 2  # packed bytes per K tile (2 codes/byte)
    SCALE_K: tl.constexpr = BLOCK_K // GROUP_SIZE  # scales per K tile
    offs_kp = tl.arange(0, PACK_K)
    offs_ks = tl.arange(0, SCALE_K)

    # A tile (BLOCK_M, PACK_K); B^T tile (PACK_K, BLOCK_N) (B stored (N, K//2)).
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_kp[None, :] * stride_ak
    b_ptrs = b_ptr + offs_kp[:, None] * stride_bk + offs_n[None, :] * stride_bn
    # Scale tiles: A (BLOCK_M, SCALE_K), B (BLOCK_N, SCALE_K) (do NOT transpose B's).
    a_scale_ptrs = a_scale_ptr + offs_m[:, None] * stride_asm + offs_ks[None, :] * stride_ask
    b_scale_ptrs = b_scale_ptr + offs_n[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk

    k_packed = K // 2
    k_scale = K // GROUP_SIZE

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        kp = k0 * PACK_K + offs_kp
        ks = k0 * SCALE_K + offs_ks
        a = tl.load(a_ptrs, mask=kp[None, :] < k_packed, other=0)
        b = tl.load(b_ptrs, mask=kp[:, None] < k_packed, other=0)
        a_scale = tl.load(a_scale_ptrs, mask=ks[None, :] < k_scale, other=0.)  # float fill casts to uint8 (e8m0) or fp8e4nv (e4m3)
        b_scale = tl.load(b_scale_ptrs, mask=ks[None, :] < k_scale, other=0.)  # b is already (PACK_K, BLOCK_N) (transposed via its pointer layout).
        acc = tl.dot_scaled(a, a_scale, 'e2m1', b, b_scale, 'e2m1', acc=acc)  # dot_scaled infers format+group from the scale dtype: uint8=E8M0/32, fp8e4nv=E4M3/16.

        a_ptrs += PACK_K * stride_ak
        b_ptrs += PACK_K * stride_bk
        a_scale_ptrs += SCALE_K * stride_ask
        b_scale_ptrs += SCALE_K * stride_bsk

    alpha = tl.load(alpha_ptr)  # 0-d fp32 scalar tensor (full precision; a python-float kernel arg would truncate)
    acc = acc * alpha
    # Store C with bounds masking; tl.store autocasts the fp32 acc to C's element type.
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def _fp4_matmul_dequant(
        a: torch.Tensor,
        b: torch.Tensor,
        a_scale: torch.Tensor,
        b_scale: torch.Tensor,
        alpha: torch.Tensor | float = 1.,
        *,
        out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    FP4 matmul C = alpha * (A @ B.T) via dequant -> cuBLAS. Used on pre-Blackwell, where tl.dot_scaled can't do
    NVFP4 (e4m3) at all and is ~3x slower than this for prefill on both formats. Dequantizes both operands to
    out_dtype with xxfp4.dequantize (which handles e8m0 AND e4m3 scales, and any N/K -- no alignment limit) and
    does one torch matmul.

    sqrt(alpha) is folded into each operand's global_scale, so da @ db.T == alpha * (A @ B.T) directly.

    Inputs:
      a, b:             (M, K//2) / (N, K//2) packed E2M1 codes (float4_e2m1fn_x2 or uint8).
      a_scale, b_scale: row-major (M/N, K//group) block scales in their NATIVE dtype (E8M0 -> mxfp4 / group 32,
                        E4M3 -> nvfp4 / group 16); dequantize infers the group from the shapes.
      alpha:            float output scale; fold any NVFP4 global scale into it (same contract as fp4_matmul).
                        Folded as sqrt(|alpha|) into each operand with alpha's sign on one of them; 0 -> zeros.
      out_dtype:        C element type (bfloat16 / float16 / float32).
    Returns: (M, N) tensor in out_dtype.
    """
    alpha = torch.as_tensor(alpha, dtype=torch.float32, device=a.device)
    mag: torch.Tensor = alpha.abs() ** .5
    da: torch.Tensor = dequantize(packed=a, scale=a_scale, global_scale=mag, dtype=out_dtype)  # (M, K)  * sqrt|alpha|
    db: torch.Tensor = dequantize(packed=b, scale=b_scale, global_scale=mag * alpha.sign(), dtype=out_dtype)  # (N, K)  * sign(alpha)*sqrt|alpha|
    return da @ db.transpose(-2, -1)  # cuBLAS GEMM in out_dtype (bf16/fp16 -> tensor cores; fp32 -> fp32 path), returns out_dtype


def fp4_matmul(
        a: torch.Tensor,
        b: torch.Tensor,
        a_scale: torch.Tensor,
        b_scale: torch.Tensor,
        alpha: torch.Tensor | float,
        *,
        out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    FP4 matmul C = alpha * (A @ B.T) -- the DECODE path (pointer loads, best at small M). Uses tl.dot_scaled,
    except NVFP4 (e4m3 scales) on pre-Blackwell, which can't (see fp4_blockscale_native) and instead routes to
    _fp4_matmul_dequant (dequant -> cuBLAS, same result). For PREFILL (large M) use fp4_matmul_tma on Blackwell,
    or _fp4_matmul_dequant on pre-Blackwell (fp4_matmul_auto picks for you).

    Inputs:
      a, b:             (M, K//2) / (N, K//2) packed E2M1 codes (float4_e2m1fn_x2 or uint8).
      a_scale, b_scale: row-major (M, K//group) / (N, K//group) block scales; the format is inferred from the
                        dtype -- E8M0 (float8_e8m0fnu) -> group 32 (mxfp4), E4M3 (float8_e4m3fn) -> group 16 (nvfp4).
      alpha:            float output scale; fold any NVFP4 global scale into it.
      out_dtype:        C element type (bfloat16 / float16 / float32).
    Returns: (M, N) tensor in out_dtype.
    """
    assert a_scale.dtype == b_scale.dtype and a_scale.dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn), a_scale.dtype
    assert out_dtype in (torch.bfloat16, torch.float16, torch.float32), f'out_dtype must be bfloat16/float16/float32, got {out_dtype}'
    M, packed_k = a.shape
    N: int = b.size(0)
    assert b.size(1) == packed_k, 'a and b must share K'
    K: int = packed_k * 2
    group_size: int = 32 if a_scale.dtype == torch.float8_e8m0fnu else 16
    assert K % group_size == 0
    assert a_scale.shape == (M, K // group_size), (a_scale.shape, (M, K // group_size))
    assert b_scale.shape == (N, K // group_size), (b_scale.shape, (N, K // group_size))
    device: torch.device = a.device
    device_index: int = device.index if device.index is not None else torch.cuda.current_device()
    # NVFP4 (e4m3 scales) on pre-Blackwell can't go through tl.dot_scaled (its upcast fallback only handles
    # e8m0 scales -- see fp4_blockscale_native), so dequant -> cuBLAS there. MXFP4 (e8m0) and NVFP4-on-Blackwell
    # use tl.dot_scaled below.
    if group_size == 16 and not fp4_blockscale_native(device_index):
        return _fp4_matmul_dequant(a, b, a_scale, b_scale, alpha, out_dtype=out_dtype)  # native-dtype scales (no uint8 view)
    if a_scale.dtype == torch.float8_e8m0fnu:  # e8m0 -> uint8 for dot_scaled; e4m3 stays fp8e4nv (Blackwell)
        a_scale, b_scale = a_scale.view(dtype=torch.uint8), b_scale.view(dtype=torch.uint8)
    c: torch.Tensor = torch.empty(M, N, device=device, dtype=out_dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    with device_guard(device):
        _fp4_matmul_kernel[grid](
            a.view(dtype=torch.uint8), b.view(dtype=torch.uint8), a_scale, b_scale, c,
            torch.as_tensor(alpha, dtype=torch.float32, device=device),
            M=M, N=N, K=K,
            stride_am=a.stride(0), stride_ak=a.stride(1), stride_bn=b.stride(0), stride_bk=b.stride(1),
            stride_asm=a_scale.stride(0), stride_ask=a_scale.stride(1), stride_bsn=b_scale.stride(0), stride_bsk=b_scale.stride(1),
            stride_cm=c.stride(0), stride_cn=c.stride(1),
            GROUP_SIZE=group_size,
        )
    return c


# -----------------------------------------------------------------------------
# 3. matmul: TMA loads + swizzled scales  (compute-bound / PREFILL)
# -----------------------------------------------------------------------------
# WHAT TMA IS: the Tensor Memory Accelerator is a Blackwell/Hopper hardware unit
# that bulk-copies a whole BLOCK-shaped tile of a tensor from global memory into
# shared memory in one asynchronous instruction, driven by a *descriptor* that
# records the tensor's base address / shape / strides. Triton exposes it as
# `TensorDescriptor.from_tensor(t, block_shape)` on the host and `desc.load([offs])`
# inside the kernel. Feeding the FP4 tensor cores this way -- instead of the
# per-element pointer arithmetic the dot_scaled kernel uses -- is what keeps them
# busy when the matmul is compute-bound (large M).
#
# SWIZZLED SCALES: unlike section 2, the scales here are NOT row-major. They use
# the cuBLAS/CUTLASS "block-scaled" swizzle (see to_blocked in section 4) -- the exact
# byte layout the tensor cores read, so our inputs match what CUTLASS consumes bit-for-bit.
# The swizzle rearranges the (rows, K/group) scale matrix in 128-row x 4-col tiles
# of BYTES; within one tile the byte at (row r, col c) moves to linear offset
#
#       (r % 32) * 16  +  (r // 32) * 4  +  c
#
# i.e. each 128x4 = 512-byte tile is stored as a (r%32, r//32, c) = (32, 4, 4)
# block. The descriptor loads a whole output tile's worth of scales as a 5-D slab
# (1, REP_M, REP_K, 2, 256), where REP_M = BLOCK_M/128 and REP_K = BLOCK_K/group/4
# count the swizzle tiles and the trailing 2 * 256 = 512 bytes is one such tile;
# the kernel reverses the permutation with
#
#       reshape(REP_M, REP_K, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(BLOCK, K/grp)
#
# Here the "32" is 128/4 (the swizzle's row-fold) and is NOT the scale group size;
# only GROUP_SIZE, REP_K and the final column count depend on the format.
#
# Descriptors bake in BLOCK_M/N/K. GROUP_M (the _grouped_pid band height; a large value == plain
# column-major) is autotuned alongside num_warps / num_stages, because grouped and column-major each win on
# different shapes.
_MATMUL_TMA_CONFIGS = [
    triton.Config({'GROUP_M': gm}, num_warps=w, num_stages=s)
    for gm, w, s in itertools.product((8, 64), (4, 8), (2, 3, 4))
]


@triton.autotune(configs=_MATMUL_TMA_CONFIGS, key=['M', 'N', 'K', 'BLOCK_M', 'BLOCK_N', 'BLOCK_K', 'GROUP_SIZE'])
@triton.jit
def _fp4_matmul_tma_kernel(
        a_desc, a_scale_desc, b_desc, b_scale_desc, c_desc, alpha_ptr,
        M, N, K,
        GROUP_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
):
    """
    C = alpha * (A @ B.T), MXFP4/NVFP4, TMA loads + swizzled scales.

    GROUP_SIZE is the scale group (32 MXFP4 / 16 NVFP4); the scale's dtype (uint8 E8M0 / float8e4nv E4M3) tells
    tl.dot_scaled the format. The swizzle is byte-level/group-independent, so the (32, 4, 4) un-swizzle is the
    same for both formats.
    """
    # No masks needed: TMA descriptors bounds-check against the full tensor shape (out-of-bounds loads zero-pad,
    # out-of-bounds stores are dropped), so partial M/N edge tiles are handled in hardware; K is a multiple of
    # BLOCK_K (asserted in fp4_matmul_tma), so the K loop has no tail either.
    REP_M: tl.constexpr = BLOCK_M // 128  # swizzle tiles per output tile, derived from the block sizes + group
    REP_N: tl.constexpr = BLOCK_N // 128
    REP_K: tl.constexpr = BLOCK_K // GROUP_SIZE // 4

    pid_m, pid_n = _grouped_pid(M, N, BLOCK_M, BLOCK_N, GROUP_M)
    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N
    offs_scale_m = pid_m * REP_M
    offs_scale_n = pid_n * REP_N
    offs_k = 0
    offs_scale_k = 0

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in tl.range(0, tl.cdiv(K, BLOCK_K)):
        a = a_desc.load([offs_am, offs_k])  # (BLOCK_M, BLOCK_K//2)
        b = b_desc.load([offs_bn, offs_k])  # (BLOCK_N, BLOCK_K//2)
        scale_a = a_scale_desc.load([0, offs_scale_m, offs_scale_k, 0, 0])  # (1, REP_M, REP_K, 2, 256) swizzled-scale slab
        scale_b = b_scale_desc.load([0, offs_scale_n, offs_scale_k, 0, 0])  # (1, REP_N, REP_K, 2, 256)

        # Un-swizzle the block-scaled layout back to per-row (BLOCK, BLOCK_K//GROUP_SIZE).
        scale_a = scale_a.reshape(REP_M, REP_K, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(BLOCK_M, BLOCK_K // GROUP_SIZE)
        scale_b = scale_b.reshape(REP_N, REP_K, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(BLOCK_N, BLOCK_K // GROUP_SIZE)
        # b is (BLOCK_N, PACK_K) from the descriptor; transpose to (PACK_K, BLOCK_N). (Unlike the pointer path's
        # strided/transposed gather, TMA copies the block as stored and the transpose is non-contiguous, so it
        # can't be a descriptor source -- hence the in-kernel transpose.)
        acc = tl.dot_scaled(a, scale_a, 'e2m1', b.trans(), scale_b, 'e2m1', acc=acc)

        offs_k += BLOCK_K // 2
        offs_scale_k += REP_K

    alpha = tl.load(alpha_ptr)  # 0-d fp32 scalar tensor (see _fp4_matmul_kernel)
    acc = acc * alpha
    # The TMA store autocasts the fp32 acc to the descriptor's element type
    # (bf16 / fp16 / fp32), so the output dtype is whatever c was allocated as.
    c_desc.store([offs_am, offs_bn], acc)


def _tma_launch(
        a: torch.Tensor, b: torch.Tensor, a_scale: torch.Tensor, b_scale: torch.Tensor, c: torch.Tensor,
        alpha: torch.Tensor | float,
        M: int, N: int, K: int,
        group_size: int,
        tile: tuple,
) -> None:
    """
    Build descriptors for one tile and launch the TMA kernel into c.
    """
    block_m, block_n, block_k = tile
    rep_m, rep_n, rep_k = block_m // 128, block_n // 128, block_k // group_size // 4

    a_desc: TensorDescriptor = TensorDescriptor.from_tensor(a.view(dtype=torch.uint8), [block_m, block_k // 2])
    b_desc: TensorDescriptor = TensorDescriptor.from_tensor(b.view(dtype=torch.uint8), [block_n, block_k // 2])
    c_desc: TensorDescriptor = TensorDescriptor.from_tensor(c, [block_m, block_n])
    a_scale_desc: TensorDescriptor = TensorDescriptor.from_tensor(a_scale, [1, rep_m, rep_k, 2, 256])
    b_scale_desc: TensorDescriptor = TensorDescriptor.from_tensor(b_scale, [1, rep_n, rep_k, 2, 256])

    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    _fp4_matmul_tma_kernel[grid](
        a_desc, a_scale_desc, b_desc, b_scale_desc, c_desc,
        torch.as_tensor(alpha, dtype=torch.float32, device=a.device),
        M=M, N=N, K=K,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, GROUP_SIZE=group_size,
    )


@functools.lru_cache(maxsize=None)  # cached per (M, N, K, group_size, out_dtype, device); the pick is shape- and device-dependent (data-independent)
def _tma_pick_tile(M: int, N: int, K: int, group_size: int, out_dtype: torch.dtype, device: torch.device) -> tuple[int, ...]:
    """
    One-time per-shape search over candidate tiles -- benchmarked on dummy tensors, since the pick depends only
    on shapes/dtypes, not data. out_dtype is part of the key: a wider C (e.g. fp32) makes the TMA store stage a
    larger shared-memory tile, so some tiles that fit a bf16/fp16 C overflow sm_120's smem and are skipped.

    Manual pick (not @triton.autotune): _tma_launch builds the TMA descriptors host-side via
    TensorDescriptor.from_tensor(t, block_shape), so BLOCK_M/N/K are fixed before launch and can't be autotune
    constexprs; autotuning the tile would need on-device tl.make_tensor_descriptor.
    """
    scale_dtype: torch.dtype = torch.float8_e4m3fn if group_size == 16 else torch.uint8  # e4m3 (nvfp4) / e8m0-as-uint8 (mxfp4)
    n_col_blocks: int = K // group_size // 4
    a: torch.Tensor = torch.empty(M, K // 2, dtype=torch.uint8, device=device)
    b: torch.Tensor = torch.empty(N, K // 2, dtype=torch.uint8, device=device)
    c: torch.Tensor = torch.empty(M, N, dtype=out_dtype, device=device)
    a_scale: torch.Tensor = torch.zeros(1, triton.cdiv(M, 128), n_col_blocks, 2, 256, dtype=scale_dtype, device=device)
    b_scale: torch.Tensor = torch.zeros(1, triton.cdiv(N, 128), n_col_blocks, 2, 256, dtype=scale_dtype, device=device)
    # Candidate tiles that fit sm_120's ~100KB smem (the tutorial's 256-wide / 4-stage tile overflows it);
    # benchmark those that divide K and keep the fastest. (128, 128, 128) is the compact fallback: when K is a
    # multiple of 128 but not 256 (e.g. odd-K padded to K_pad), it is the ONLY BLOCK_K-divisible tile whose store
    # stage still fits an fp32 C -- the wider 128x256 / 256x128 tiles need 128KB smem and OutOfResources on fp32.
    best_ms, best_tile = torch.inf, None
    for cand in [(128, 128, 128), (128, 128, 256), (128, 256, 128), (128, 256, 256), (256, 128, 128), (256, 128, 256)]:
        block_k: int = cand[2]
        if K % block_k != 0 or block_k % group_size != 0 or (block_k // group_size) % 4 != 0:
            continue  # BLOCK_K must split into whole swizzle tiles (rep_k = block_k/group_size/4)
        try:
            ms: float = triton.testing.do_bench(
                lambda t=cand: _tma_launch(a=a, b=b, a_scale=a_scale, b_scale=b_scale, c=c, alpha=1., M=M, N=N, K=K, group_size=group_size, tile=t),
                warmup=10,
                rep=30,
            )
        except Exception:
            continue
        if ms < best_ms:
            best_ms, best_tile = ms, cand
    if best_tile is None:
        raise RuntimeError(f'no TMA tile fits for shape (M={M}, N={N}, K={K}, group_size={group_size}, {out_dtype})')
    return best_tile


def fp4_matmul_tma(
        a: torch.Tensor,
        b: torch.Tensor,
        a_scale_blocked: torch.Tensor,
        b_scale_blocked: torch.Tensor,
        alpha: torch.Tensor | float,
        *,
        out_dtype: torch.dtype = torch.bfloat16,
        tile: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """
    FP4 matmul C = alpha * (A @ B.T) via TMA + swizzled scales -- the PREFILL path (best at large M).

    Inputs:
      a, b:           (M, K//2) / (N, K//2) packed E2M1 codes (float4_e2m1fn_x2 or uint8).
      a_scale_blocked, b_scale_blocked: 1-D to_blocked swizzle of the (M, K//group) / (N, K//group) row-major
                        scales -- flat, ceil(rows/128) * ceil((K//group)/4) * 512 bytes (as CUTLASS consumes); the
                        format is inferred from the dtype -- E8M0 (float8_e8m0fnu) -> mxfp4, E4M3 (float8_e4m3fn) -> nvfp4.
      alpha:          float output scale; fold any NVFP4 global scale into it.
      out_dtype:      C element type (bfloat16 / float16 / float32).
      tile:           (BLOCK_M, BLOCK_N, BLOCK_K) override; defaults to the best per-shape pick.
    Returns: (M, N) tensor in out_dtype.
    """
    assert a_scale_blocked.dtype == b_scale_blocked.dtype and a_scale_blocked.dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn), a_scale_blocked.dtype
    assert out_dtype in (torch.bfloat16, torch.float16, torch.float32), f'out_dtype must be bfloat16/float16/float32, got {out_dtype}'
    M, packed_k = a.shape
    N: int = b.size(0)
    assert b.size(1) == packed_k, 'a and b must share K'
    K: int = packed_k * 2
    group_size: int = 32 if a_scale_blocked.dtype == torch.float8_e8m0fnu else 16
    if group_size == 16:  # NVFP4 via TMA feeds tl.dot_scaled e4m3 scales -> only the native FP4 block-scale MMA handles it
        assert fp4_blockscale_native(a.device.index if a.device.index is not None else torch.cuda.current_device()), \
            'NVFP4 via TMA requires Blackwell (sm_100+); use fp4_matmul / fp4_matmul_auto (dequant -> cuBLAS) on older arches'
    # K must split into whole 4-wide swizzle tiles, else the reshape below mismatches to_blocked's ceil-padding.
    assert K % group_size == 0 and K // group_size % 4 == 0, f'TMA path needs K % {group_size * 4} == 0 (got K={K})'
    # TMA C-store: the descriptor's row stride is N * out_dtype.itemsize bytes and must be 16-byte aligned, so N
    # must be a multiple of 16 // itemsize (8 for bf16/fp16, 4 for fp32). Unaligned N can't use TMA.
    n_align: int = 16 // out_dtype.itemsize
    assert N % n_align == 0, f'TMA path needs N % {n_align} == 0 for out_dtype={out_dtype} (C-store row stride N*{out_dtype.itemsize}B must be 16B-aligned); got N={N} -- use fp4_matmul for unaligned N'
    n_col_blocks: int = K // group_size // 4  # swizzle tiles along K
    if a_scale_blocked.dtype == torch.float8_e8m0fnu:  # only e8m0 -> uint8; e4m3 stays native (dot_scaled infers both)
        a_scale_blocked, b_scale_blocked = a_scale_blocked.view(dtype=torch.uint8), b_scale_blocked.view(dtype=torch.uint8)
    a_scale: torch.Tensor = a_scale_blocked.reshape(1, triton.cdiv(M, 128), n_col_blocks, 2, 256)
    b_scale: torch.Tensor = b_scale_blocked.reshape(1, triton.cdiv(N, 128), n_col_blocks, 2, 256)
    device: torch.device = a.device
    c: torch.Tensor = torch.empty(M, N, dtype=out_dtype, device=device)
    with device_guard(device):
        if tile is None:
            tile = _tma_pick_tile(M=M, N=N, K=K, group_size=group_size, out_dtype=c.dtype, device=device)
        assert K % tile[2] == 0, f'K={K} not a multiple of BLOCK_K={tile[2]}'
        _tma_launch(
            a=a, b=b, a_scale=a_scale, b_scale=b_scale, c=c,
            alpha=alpha,
            M=M, N=N, K=K,
            group_size=group_size,
            tile=tile,
        )
    return c

# -----------------------------------------------------------------------------
# 4. Scale swizzle (to_blocked) + K-padding helpers, and the arch/shape auto-router
# -----------------------------------------------------------------------------

@triton.jit
def _scale_swizzle_kernel(
        scale_ptr, out_ptr,
        n_rows, n_cols,
        in_row_stride, n_col_blocks,
        NCB: tl.constexpr,
):
    """
    Coalesced cuBLAS/CUTLASS block-scale swizzle. One program = one 128-row block x NCB 4-col swizzle tiles.
    GATHER the scattered source bytes and write each 512-byte (128x4) tile CONTIGUOUSLY -- the inverse of the
    dest = (r%32)*16 + (r//32)*4 + c scatter -- so the global stores coalesce (vs a stride-16 scatter store).
    """
    pid_row = tl.program_id(0)  # 128-row block
    pid_cb = tl.program_id(1)   # group of NCB 4-col blocks
    o = tl.arange(0, 512)       # linear offset within a 128x4 swizzled tile (128 * 4)
    r = (o % 16 // 4) * 32 + o // 16  # invert dest=(r%32)*16+(r//32)*4+c -> (r, c)
    c = o % 4
    cb = pid_cb * NCB + tl.arange(0, NCB)  # this program's column blocks
    gr = pid_row * 128 + r[None, :]
    gc = cb[:, None] * 4 + c[None, :]
    in_bounds = cb[:, None] < n_col_blocks
    x = tl.load(scale_ptr + gr * in_row_stride + gc, mask=in_bounds & (gr < n_rows) & (gc < n_cols), other=0)
    dst = pid_row * (512 * n_col_blocks) + cb[:, None] * 512 + o[None, :]  # contiguous within each 512-byte tile
    tl.store(out_ptr + dst, x, mask=in_bounds)


def to_blocked_torch(scale: torch.Tensor) -> torch.Tensor:
    """
    Pure-PyTorch cuBLAS/CUTLASS block-scale swizzle -- the byte layout the FP4 tensor cores read (the inverse
    of the TMA kernel's un-swizzle in section 3). Zero-pad (rows, cols) to (ceil(rows/128)*128, ceil(cols/4)*4),
    then lay each 128x4 byte tile out as (32, 4, 4): byte (r, c) -> (r % 32) * 16 + (r // 32) * 4 + c. Flat out.
    Reference implementation: device-agnostic, and the oracle to_blocked is checked byte-identical against.
    """
    rows, cols = scale.shape
    nrb, ncb = -(-rows // 128), -(-cols // 4)  # ceil-div: count of 128-row x 4-col swizzle tiles
    padded: torch.Tensor = torch.nn.functional.pad(scale.view(dtype=torch.uint8), (0, ncb * 4 - cols, 0, nrb * 128 - rows))  # zero-pad right/bottom to whole 128x4 tiles
    # split rows 128 -> (r//32 = 4, r%32 = 32) and cols -> (ncb, 4 = c), then transpose to tile order (r%32, r//32, c)
    rearranged: torch.Tensor = padded.unflatten(dim=0, sizes=(nrb, 4, 32)).unflatten(dim=-1, sizes=(ncb, 4)).transpose(1, 3)  # (nrb, ncb, r%32, r//32, c)
    return rearranged.flatten().view(dtype=scale.dtype)


def to_blocked(scale: torch.Tensor) -> torch.Tensor:
    """
    cuBLAS/CUTLASS block-scale swizzle (the byte layout the FP4 tensor cores read) -- a fast coalesced Triton
    kernel on CUDA, falling back to the pure-torch reference (to_blocked_torch) off-CUDA. Byte-identical to
    to_blocked_torch (asserted in _unit_test); ~1.3-1.7x faster than the torch / qutlass-naive swizzles.
    Input: contiguous row-major (rows, cols) fp8 scale; output: the flat swizzled scale in the input dtype.
    """
    device: torch.device = scale.device
    if device.type != 'cuda':
        return to_blocked_torch(scale)
    rows, cols = scale.shape
    nrb, ncb = triton.cdiv(rows, 128), triton.cdiv(cols, 4)  # count of 128-row x 4-col swizzle tiles
    out: torch.Tensor = torch.empty(nrb * 128, ncb * 4, dtype=scale.dtype, device=device)  # padded to whole 128x4 SF tiles, native fp8 dtype
    su8: torch.Tensor = scale.view(dtype=torch.uint8)  # rows are contiguous -> stride(0) is the row stride; cols unit-stride
    with device_guard(device):  # launch on the scale's device (multi-GPU correctness)
        _scale_swizzle_kernel[(nrb, triton.cdiv(ncb, 8))](
            su8, out.view(dtype=torch.uint8),
            n_rows=rows, n_cols=cols,
            in_row_stride=su8.stride(0), n_col_blocks=ncb,
            NCB=8,
        )
    return out.flatten()


def pad_k_to_tile(
        a: torch.Tensor,
        b: torch.Tensor,
        a_scale: torch.Tensor,
        b_scale: torch.Tensor,
        group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Zero-pad the contraction dim K of the packed operands + row-major block scales up to a multiple of 128, so
    the block-scaled matmuls (TMA / qutlass) accept a K that is a multiple of group_size but not of group_size*4
    (a whole 128x4 swizzle tile). E2M1 code 0x00 = +0.0 and the padded-group scales multiply those zeros, so the
    padding adds nothing to the dot product -- numerically inert. No-op (returns the inputs unchanged) when K is
    already 128-aligned. Returns ROW-MAJOR scales; the caller applies to_blocked afterward.

    Inputs:
      a, b:             (M, K//2) / (N, K//2) packed E2M1 codes (float4_e2m1fn_x2 or uint8).
      a_scale, b_scale: row-major (M, K//group_size) / (N, K//group_size) block scales (native fp8 dtype).
      group_size:       32 (mxfp4) / 16 (nvfp4).
    Returns: the padded (a, b, a_scale, b_scale) -- IN THAT ORDER (unpack to match, else operands/scales swap).
    """
    k: int = a.size(-1) * 2  # packed last dim is K//2
    k_pad: int = -(-k // 128) * 128  # ceil(k/128)*128
    if k_pad == k:
        return a, b, a_scale, b_scale  # already 128-aligned: pass the originals through, no view round-trip
    pad_bytes: int = (k_pad - k) // 2  # packed K//2 -> K_pad//2 (K is a multiple of group_size, so this is whole)
    pad_cols: int = (k_pad - k) // group_size  # scale K//group -> K_pad//group
    pad_packed = lambda t: torch.nn.functional.pad(t.view(dtype=torch.uint8), (0, pad_bytes)).view(dtype=t.dtype)  # right-pad the last dim with zeros
    pad_scale = lambda t: torch.nn.functional.pad(t.view(dtype=torch.uint8), (0, pad_cols)).view(dtype=t.dtype)
    return pad_packed(a), pad_packed(b), pad_scale(a_scale), pad_scale(b_scale)


def fp4_matmul_auto(
        a: torch.Tensor,
        b: torch.Tensor,
        a_scale: torch.Tensor,
        b_scale: torch.Tensor,
        alpha: torch.Tensor | float,
        *,
        out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    FP4 matmul C = alpha * (A @ B.T), auto-routing by arch + M. Takes ROW-MAJOR scales (like fp4_matmul).
      * Blackwell (native FP4 MMA): pointer path for decode (small M), TMA for prefill (large M) -- swizzling
        scales with to_blocked on the fly (and zero-padding K up to a multiple of 128 via pad_k_to_tile) --
        falling back to the pointer path only when N is not (16//itemsize)-aligned.
      * Pre-Blackwell (TMA dead): dequant -> cuBLAS (_fp4_matmul_dequant) for prefill on BOTH formats (it beats
        dot_scaled ~3x there, and is the only NVFP4 option); pointer path (fp4_matmul) for decode.
    Accepts any shape fp4_matmul does.

    Inputs:
      a, b:             (M, K//2) / (N, K//2) packed E2M1 codes (float4_e2m1fn_x2 or uint8).
      a_scale, b_scale: row-major (M, K//group) / (N, K//group) block scales; format inferred from the dtype
                        (E8M0 (float8_e8m0fnu) -> mxfp4 / group 32, E4M3 (float8_e4m3fn) -> nvfp4 / group 16).
      alpha:            float output scale; fold any NVFP4 global scale into it.
      out_dtype:        C element type (bfloat16 / float16 / float32).
    Returns: (M, N) tensor in out_dtype.
    """
    M: int = a.size(0)
    N: int = b.size(0)
    group_size: int = 32 if a_scale.dtype == torch.float8_e8m0fnu else 16
    di: int = a.device.index if a.device.index is not None else torch.cuda.current_device()
    blackwell: bool = fp4_blockscale_native(di)
    prefill: bool = M >= 128  # prefill threshold: TMA / dequant decisively ahead by M=256; pointer path ahead at M<=16 (benchmark)
    tma_shape_ok: bool = N % (16 // out_dtype.itemsize) == 0  # K need not be (group*4)-aligned: pad_k_to_tile pads it to a multiple of 128 below
    # The PREFILL fast path is TMA on Blackwell, dequant -> cuBLAS pre-Blackwell (TMA is dead / can't compile there).
    # Can this arch's prefill path take this shape? Pre-Blackwell dequant takes ANY shape; Blackwell TMA needs only
    # aligned N -- a K that isn't a whole 128-swizzle tile is zero-padded up to a multiple of 128 (fp4 0x00 = +0.0
    # is inert). A Blackwell shape with unaligned N, or any decode (small M), falls to the pointer path (fp4_matmul).
    if prefill and blackwell and tma_shape_ok:
        a, b, a_scale, b_scale = pad_k_to_tile(a=a, b=b, a_scale=a_scale, b_scale=b_scale, group_size=group_size)  # pad BEFORE to_blocked; unpack order matches the return
        return fp4_matmul_tma(a, b, to_blocked(a_scale), to_blocked(b_scale), alpha, out_dtype=out_dtype)
    if prefill and not blackwell:
        return _fp4_matmul_dequant(a, b, a_scale, b_scale, alpha, out_dtype=out_dtype)
    return fp4_matmul(a, b, a_scale, b_scale, alpha, out_dtype=out_dtype)


# -----------------------------------------------------------------------------
# 5. Correctness + benchmark harness
# -----------------------------------------------------------------------------

def _unit_test(
        device: torch.device = torch.device('cuda'),
) -> None:
    """
    Unit test: each FP4 matmul path (direct pointer loads, dequant->cuBLAS, TMA) for mxfp4 / nvfp4 vs a float64
    dequant reference. Operands are quantized with the pure-PyTorch quantize_fp4.rtn_xxfp4 and the reference is
    quantize_fp4.dequant_xxfp4(...) @ .T. Two tolerance classes (see close()): tight element-wise allclose for
    the exact-operand dot_scaled / TMA paths, peak-relative for the bf16/fp16 dequant GEMM. Covers skinny /
    partial-M / non-power-of-two-N bf16 shapes plus fp16 / fp32 output dtypes. (The TMA / NVFP4-Blackwell
    variants are skipped on pre-Blackwell, where NVFP4/direct itself uses the dequant path.)
    """

    from quantize_fp4 import rtn_xxfp4, dequant_xxfp4

    torch.manual_seed(seed=0)
    di: int = device.index if device.index is not None else torch.cuda.current_device()
    blackwell: bool = fp4_blockscale_native(di)  # NVFP4 via dot_scaled (direct on Blackwell + the TMA path) needs it; pre-Blackwell NVFP4 dequants (direct == the dequant variant) and NVFP4/TMA is skipped
    fmt_cfg: dict[str, dict] = {
        'mxfp4': {'group_size': 32, 'scale_dtype': torch.float8_e8m0fnu, 'scale_scale': 4., 'global_scale': 1. / 3.},
        'nvfp4': {'group_size': 16, 'scale_dtype': torch.float8_e4m3fn, 'scale_scale': 6., 'global_scale': .1},
    }
    n_checks, failures = 0, []

    def check(ok: bool, tag: str, detail: str) -> None:
        nonlocal n_checks
        n_checks += 1
        print(f"{'PASS' if ok else 'FAIL'} {tag}  ({detail})")
        if not ok:
            failures.append(tag)

    def close(out: torch.Tensor, ref: torch.Tensor, dt: torch.dtype, dequant_path: bool) -> tuple[bool, float]:
        """
        (dtype-ok AND numerically-close, max|out-ref|). dot_scaled / TMA keep operands exact and accumulate in
        fp32, so a tight ELEMENT-WISE allclose holds. The dequant path is a plain bf16/fp16 cuBLAS GEMM of the
        dequantized values: its floor is ~2**-8 (bf16) / 2**-11 (fp16) RELATIVE TO THE PEAK -- element-wise rtol is
        meaningless at cancellation cells (small |ref|, error set by the row/col magnitudes).
        """
        out64: torch.Tensor = out.to(dtype=torch.float64)
        max_err: float = (out64 - ref).abs().max().item()
        if dequant_path:
            peak_rtol: float = 1.2e-2 if dt is torch.bfloat16 else 1.5e-3  # fp32-out dequant is near-exact; bf16/fp16 are GEMM-floor
            ok: bool = out64.allclose(ref, rtol=0., atol=peak_rtol * ref.abs().max().item() + 2e-2)
        else:
            rtol: float = 6e-3 if dt is torch.bfloat16 else 1e-3
            ok = out64.allclose(ref, rtol=rtol, atol=2e-2)
        return ok and out.dtype == dt, max_err

    def quantized(fmt: str, a: torch.Tensor, b: torch.Tensor):
        """
        Quantize a and b for fmt (pure-PyTorch); return packed a/b, the float64 dequant(a) @ dequant(b).T
        reference, the per-format alpha (= gs**2), and the (name, fn, a_scale, b_scale) variants to run.
        """
        cfg: dict = fmt_cfg[fmt]
        gs: float = cfg['global_scale']

        def q(x: torch.Tensor):
            r: dict = rtn_xxfp4(x=x, group_size=cfg['group_size'], scale_dtype=cfg['scale_dtype'], scale_scale=cfg['scale_scale'], global_scale=gs, fp4_rounding_mode='even')
            return r['e2m1'], r['scale_quant'], to_blocked(r['scale_quant'])

        ap, asc, ablk = q(a)
        bp, bsc, bblk = q(b)
        da: torch.Tensor = dequant_xxfp4(e2m1=ap, scale_quant=asc, global_scale=gs, dtype=torch.float64)
        db: torch.Tensor = dequant_xxfp4(e2m1=bp, scale_quant=bsc, global_scale=gs, dtype=torch.float64)
        ref: torch.Tensor = da @ db.transpose(-2, -1)
        alpha: float = gs ** 2.
        # direct: dot_scaled (mxfp4) or dequant->cuBLAS (nvfp4 pre-Blackwell). dequant: _fp4_matmul_dequant on
        # every arch/shape. Both use ROW-MAJOR (native-dtype) scales. tma: swizzled scales, Blackwell-only for nvfp4.
        variants = [('direct', fp4_matmul, asc, bsc), ('dequant', _fp4_matmul_dequant, asc, bsc)]
        if fmt == 'mxfp4' or blackwell:
            variants.append(('tma', fp4_matmul_tma, ablk, bblk))
        return ap, bp, ref, alpha, variants

    # 0) to_blocked (Triton coalesced swizzle) must be byte-identical to the pure-torch reference to_blocked_torch
    #    -- for both scale dtypes (e8m0 mxfp4 / e4m3 nvfp4: it is a pure byte swizzle, dtype only sets the output
    #    dtype) -- over row-padding (rows % 128 != 0) and col-padding (cols % 4 != 0) cases.
    for sdt in torch.float8_e8m0fnu, torch.float8_e4m3fn:
        for rr, cc in (256, 68), (256, 65), (51, 130), (4096, 256), (130, 7), (1, 1), (8192, 512):
            sc: torch.Tensor = torch.randint(0, 256, (rr, cc), dtype=torch.uint8, device=device).view(dtype=sdt)
            bt, br = to_blocked(sc), to_blocked_torch(sc)
            check(bt.dtype == sdt and bt.view(dtype=torch.uint8).equal(br.view(dtype=torch.uint8)), f"to_blocked==torch[{str(sdt).rsplit('.', 1)[-1]},{rr}x{cc}]", f'numel {bt.numel()}')

    # 1) bf16 / fp16 / fp32 output on every variant over a few even-N shapes (decode / prefill+partial-M+non-pow2-N).
    #    Inputs *2 keep fp16 in range. Tolerances per the path's precision class (see close()): tight element-wise
    #    for dot_scaled / TMA (exact operands), peak-relative GEMM floor for the dequant path.
    def dequant_path(name: str, fmt: str) -> bool:  # which variants are a bf16/fp16 dequant->cuBLAS GEMM
        return name == 'dequant' or (name == 'direct' and fmt == 'nvfp4' and not blackwell)  # nvfp4/direct dequants pre-Blackwell

    for m, n, k in (16, 4096, 4096), (1027, 2080, 2048):
        a: torch.Tensor = torch.randn(m, k, dtype=torch.bfloat16, device=device) * 2.
        b: torch.Tensor = torch.randn(n, k, dtype=torch.bfloat16, device=device) * 2.
        for fmt in fmt_cfg.keys():
            ap, bp, ref, alpha, variants = quantized(fmt, a, b)
            for name, fn, sa, sb in variants:
                for dt in torch.bfloat16, torch.float16, torch.float32:
                    out: torch.Tensor = fn(ap, bp, sa, sb, alpha, out_dtype=dt)
                    ok, max_err = close(out, ref, dt, dequant_path(name, fmt))
                    check(ok, f"{fmt}/{name}[m={m},n={n},k={k},out={str(dt).rsplit('.', 1)[-1]}]", f'dtype {out.dtype}, max|out-ref| {max_err:.3e}')

    # 2) Odd N (unaligned C-store row stride): the pointer + dequant paths stay correct (every out dtype);
    #    TMA must reject it.
    for m, n, k in (33, 513, 1024),:
        a: torch.Tensor = torch.randn(m, k, dtype=torch.bfloat16, device=device) * 2.
        b: torch.Tensor = torch.randn(n, k, dtype=torch.bfloat16, device=device) * 2.
        for fmt in fmt_cfg.keys():
            ap, bp, ref, alpha, variants = quantized(fmt, a, b)
            for name, fn, sa, sb in variants:
                if name == 'tma':  # TMA rejects odd N (no masking); direct + dequant must handle it
                    try:
                        fn(ap, bp, sa, sb, alpha)
                        tma_raised: bool = False
                    except (AssertionError, RuntimeError):
                        tma_raised = True
                    check(tma_raised, f'{fmt}/tma[m={m},n={n},k={k}] odd-N rejected', 'TMA needs N % 8 == 0 (bf16); raises')
                    continue
                for dt in torch.bfloat16, torch.float16, torch.float32:
                    out: torch.Tensor = fn(ap, bp, sa, sb, alpha, out_dtype=dt)
                    ok, max_err = close(out, ref, dt, dequant_path(name, fmt))
                    check(ok, f"{fmt}/{name}[m={m},n={n},k={k},out={str(dt).rsplit('.', 1)[-1]}]", f'dtype {out.dtype}, max|out-ref| {max_err:.3e}')

    # 3) Odd K (K % group == 0 but K % (group*4) != 0): pad_k_to_tile zero-pads K up to a multiple of 128 for the
    #    block-scaled TMA / qutlass paths (fp4 0x00 = +0.0 is inert). K=2080 is a multiple of both group sizes
    #    (32, 16) but NOT of group*4 (128 / 64), so the unpadded TMA rejects it today; K_pad = 2176.
    for m, n, k in (256, 512, 2080),:
        assert k % 128 != 0 and k % 32 == 0 and k % 16 == 0, 'odd-K shape must be group-aligned but not 128-aligned'
        a: torch.Tensor = torch.randn(m, k, dtype=torch.bfloat16, device=device) * 2.
        b: torch.Tensor = torch.randn(n, k, dtype=torch.bfloat16, device=device) * 2.
        k_pad: int = -(-k // 128) * 128
        for fmt in fmt_cfg.keys():
            g: int = fmt_cfg[fmt]['group_size']
            ap, bp, ref, alpha, variants = quantized(fmt, a, b)
            asc, bsc = variants[0][2], variants[0][3]  # variants[0] = ('direct', fp4_matmul, asc, bsc) -- row-major scales

            # (a) pad_k_to_tile is INERT: pad K, run the POINTER path on padded vs unpadded, both vs ref (runs on
            #     every arch). close() not torch.equal: nvfp4 pointer dequants pre-Blackwell -> identical to sub-ULP.
            app, bpp, ascp, bscp = pad_k_to_tile(ap, bp, asc, bsc, g)
            check(app.size(-1) * 2 == k_pad and ascp.shape == (m, k_pad // g), f'{fmt}/pad-shape[k={k}]', f'packed K {app.size(-1) * 2}, scale {tuple(ascp.shape)}')
            for dt in torch.bfloat16, torch.float16, torch.float32:
                out_pad: torch.Tensor = fp4_matmul(app, bpp, ascp, bscp, alpha, out_dtype=dt)
                out_raw: torch.Tensor = fp4_matmul(ap, bp, asc, bsc, alpha, out_dtype=dt)
                ok_ref, e_ref = close(out_pad, ref, dt, dequant_path('direct', fmt))
                ok_eq, e_eq = close(out_pad, out_raw.to(dtype=torch.float64), dt, dequant_path('direct', fmt))
                check(ok_ref and ok_eq, f"{fmt}/pad-pointer[m={m},n={n},k={k},out={str(dt).rsplit('.', 1)[-1]}]", f'vs ref {e_ref:.3e}, vs unpadded {e_eq:.3e}')

            # (b) padded operands through fp4_matmul_tma DIRECTLY (mxfp4 runs on sm_89; nvfp4 TMA is Blackwell-only).
            #     Tight close() (exact operands, fp32 accumulate). The load-bearing TMA-padding coverage on this box.
            if fmt == 'mxfp4' or blackwell:
                for dt in torch.bfloat16, torch.float16, torch.float32:
                    out_tma: torch.Tensor = fp4_matmul_tma(app, bpp, to_blocked(ascp), to_blocked(bscp), alpha, out_dtype=dt)
                    ok, e = close(out_tma, ref, dt, False)
                    check(ok, f"{fmt}/pad-tma[m={m},n={n},k={k},out={str(dt).rsplit('.', 1)[-1]}]", f'max|out-ref| {e:.3e}')

            # (c) the UNPADDED odd-K through fp4_matmul_tma must still be rejected (guards the K assert against being
            #     loosened without going through pad_k_to_tile; nvfp4 also raises its Blackwell assert pre-Blackwell).
            try:
                fp4_matmul_tma(ap, bp, to_blocked(asc), to_blocked(bsc), alpha)
                tma_raised: bool = False
            except (AssertionError, RuntimeError):
                tma_raised = True
            check(tma_raised, f'{fmt}/tma[m={m},n={n},k={k}] unpadded odd-K rejected', f'TMA needs K % {g * 4} == 0; raises')

            # (d) end-to-end via fp4_matmul_auto: dequant -> cuBLAS pre-Blackwell, padded-TMA on Blackwell -- match ref.
            for dt in torch.bfloat16, torch.float16, torch.float32:
                out_auto: torch.Tensor = fp4_matmul_auto(ap, bp, asc, bsc, alpha, out_dtype=dt)
                ok, e = close(out_auto, ref, dt, not blackwell)
                check(ok, f"{fmt}/auto[m={m},n={n},k={k},out={str(dt).rsplit('.', 1)[-1]}]", f'max|out-ref| {e:.3e}')

    print(f"\n{'-' * 60}\n{n_checks - len(failures)}/{n_checks} passed")
    assert not failures, failures
    print('Unit test passed.')


def _benchmark(
        device: torch.device = torch.device('cuda'),
) -> None:
    """
    FP4 matmul TF/s over LLAMA linear-layer shapes x batch sizes: 'direct' (pointer dot_scaled), 'dequant' (dequant -> cuBLAS), 'tma' (TMA + swizzled scales).
    """

    fmt_cfg: dict[str, dict] = {
        'mxfp4': {'group_size': 32, 'scale_dtype': torch.float8_e8m0fnu, 'scale_scale': 4., 'global_scale': 1. / 3.},
        'nvfp4': {'group_size': 16, 'scale_dtype': torch.float8_e4m3fn, 'scale_scale': 6., 'global_scale': .1},
    }
    llama_layers: dict[str, list[tuple[int, int]]] = {  # (K, N) per linear layer, by model size
        '7B': [(4096, 3 * 4096), (4096, 4096), (4096, 2 * 10752), (10752, 4096)],
        '13B': [(5120, 3 * 5120), (5120, 5120), (5120, 2 * 13568), (13568, 5120)],
        '70B': [(8192, 3 * 8192), (8192, 8192), (8192, 2 * 21760), (21760, 8192)],
    }

    def gmean(xs: list[float]) -> float:
        return torch.as_tensor(xs).log().mean().exp().item() if xs else torch.nan

    print("'direct' = pointer dot_scaled (best for decode); 'dequant' = dequant -> cuBLAS (best for prefill pre-Blackwell);"
          "\n'tma' = TMA + swizzled scales (best for prefill on Blackwell). TF/s per path; 'best' = min latency."
          "\n('tma' n/a for NVFP4 on pre-Blackwell -> shown 0.)")
    blackwell: bool = fp4_blockscale_native(device.index if device.index is not None else torch.cuda.current_device())
    for fmt in ('mxfp4', 'nvfp4'):
        cfg: dict = fmt_cfg[fmt]
        alpha: float = cfg['global_scale'] ** 2.

        def q(rows: int, k: int):
            packed: torch.Tensor = torch.randint(0, 256, (rows, k // 2), dtype=torch.uint8, device=device)  # random packed E2M1 codes
            # O(1) random scales, NOT torch.zeros: an all-zero e8m0 scale is the denormal 2**-127, whose dequant
            # runs ~2.7x slower than normal-range scales and would under-report MXFP4 throughput.
            scale: torch.Tensor = (torch.rand(rows, k // cfg['group_size'], dtype=torch.float32, device=device) + .5).to(dtype=cfg['scale_dtype'])
            return packed, scale, to_blocked(scale)

        print(f'\n{fmt.upper()} matmul: Triton direct / dequant / tma  [{torch.cuda.get_device_name(0)}]')
        print(f"{'shape (MxNxK)':>22} | {'direct':>9} | {'dequant':>9} | {'tma':>9} | {'best TF/s':>9}")
        print('-' * 72)
        decode, prefill = [], []
        for layers in llama_layers.values():
            for k, n in layers:
                for m in 1, 16, 256, 1024:
                    ap, asc, ablk = q(m, k)
                    bp, bsc, bblk = q(n, k)
                    tf = lambda t, _f=2. * m * n * k: _f / t * 1e-9
                    t_scaled: float = triton.testing.do_bench(lambda: fp4_matmul(ap, bp, asc, bsc, alpha), warmup=10, rep=30)
                    t_dequant: float = triton.testing.do_bench(lambda: _fp4_matmul_dequant(ap, bp, asc, bsc, alpha), warmup=10, rep=30)
                    tma_avail: bool = fmt == 'mxfp4' or blackwell  # NVFP4/TMA needs Blackwell; skip it elsewhere
                    t_tma: float = triton.testing.do_bench(lambda: fp4_matmul_tma(ap, bp, ablk, bblk, alpha), warmup=10, rep=30) if tma_avail else float('inf')
                    t_best: float = min(t_scaled, t_dequant, t_tma)
                    (decode if m <= 16 else prefill).append(tf(t_best))
                    print(f"{f'{m}x{n}x{k}':>22} | {tf(t_scaled):>7.0f}  | {tf(t_dequant):>7.0f}  | {tf(t_tma):>7.0f}  | {tf(t_best):>7.0f}")
        print('-' * 72)
        print(f'{fmt.upper()} best-Triton geomean TF/s:  decode (M<=16) {gmean(decode):.0f} | prefill (M>=256) {gmean(prefill):.0f}')

    # to_blocked scale-swizzle micro-bench: the coalesced Triton kernel vs the pure-torch reference (us, lower=better).
    # Runs once per (rows, cols) scale shape (format-agnostic -- the swizzle is byte-level). Shapes ~ (M or N, K//group).
    print(f'\nto_blocked scale-swizzle  [{torch.cuda.get_device_name(0)}]')
    print(f"{'scale (rows x cols)':>22} | {'torch':>9} | {'triton':>9} | {'speedup':>9}")
    print('-' * 60)
    for rows, cols in (1024, 256), (4096, 256), (11008, 256), (4096, 1024), (8192, 512):
        s: torch.Tensor = torch.randint(0, 256, (rows, cols), dtype=torch.uint8, device=device).view(dtype=torch.float8_e8m0fnu)
        us_torch: float = triton.testing.do_bench(lambda: to_blocked_torch(s), warmup=25, rep=100) * 1e3
        us_triton: float = triton.testing.do_bench(lambda: to_blocked(s), warmup=25, rep=100) * 1e3
        print(f"{f'{rows}x{cols}':>22} | {us_torch:>7.1f}u  | {us_triton:>7.1f}u  | {us_torch / us_triton:>7.2f}x")


if __name__ == '__main__':
    _unit_test(device=torch.device('cuda'))
    _benchmark(device=torch.device('cuda'))
