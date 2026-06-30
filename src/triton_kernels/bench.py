"""
Unified performance gate for ALL triton_kernels/ kernels -- one portable, headless entry point.

Every kernel is checked against a ceiling MEASURED on the running GPU, so the gate ports to any device
without hardcoded peaks. Two ceilings are measured up front:
  * copy_ceiling()    -- cold read+write DRAM-copy bandwidth (GB/s); the bandwidth roofline.
  * matmul_ceiling()  -- a large square cuBLAS GEMM (TF/s, bf16 + fp16); the compute roofline anchor.

THE ROOFLINE CHECK (shared by every section):
    roofline_time = max( bytes_moved / copy_ceiling ,        # DRAM-bound floor
                         cublas_time_at_same_shape )          # compute-bound floor (equivalent GEMM)
    efficiency    = roofline_time / actual_time               # PASS if >= a per-section fraction
`efficiency > 1` is legitimate and is reported uncapped as a win, never a failure: a kernel beats the bf16
GEMM on Blackwell native FP4 (the 5090 hardware MMA), and the Hessian lower-only path beats the full GEMM by
exploiting symmetry (~2x). bf16 cuBLAS is the single cross-arch baseline (no portable native-FP4 cuBLAS).

THE VERDICT gates on all three sections by default: the single-pass DRAM-bound check plus the two
GPU-portable compute-equivalence checks (FP4 matmul + Hessian). Caveat: whether a single-pass kernel is DRAM-
vs compute-bound is GPU-dependent -- the heaviest fused op is compute-bound on low-FLOP GPUs (optimal, not a
regression), so the DRAM gate can FAIL there even though nothing regressed. Pass --lenient to demote the
single-pass section to a diagnostic (gate on FP4 + Hessian only), and/or use `compare` against a reference
machine to catch single-pass regressions directly.

THREE SECTIONS (all run by default; each prints a table + a section verdict):
  * bench_dram     -- single-pass kernels (transform / xxfp4 / e2m1 / fused); pure bandwidth roofline (gated; --lenient to demote).
                      transform.py : block_transform
                      xxfp4.py     : fake_quantize / quantize_pack / dequantize   (mxfp4 e8m0; + nvfp4 e4m3)
                      e2m1.py      : fake_quantize / quantize_pack / dequantize   (grid-only, no scale)
                      fused.py     : transform_fakequant / transform_quantize_pack   (mxfp4; + nvfp4)
                      The transform_quantize_pack rows ALSO race the qutlass fused quant (fusedQuantize{Mx,WushMx,
                      Nv}) when qutlass is importable -- extra `qut GB/s | speedup | match` columns (informational,
                      not gated; the occupancy-bound GT=128 row is informational so it can't fail the verdict).
  * bench_fp4      -- fp4_matmul_auto (the production router) vs the qutlass CUDA fp4 matmul (shown when
                      qutlass is importable) vs a measured bf16 cuBLAS GEMM at the same (M,N,K). Decode (small
                      M) is DRAM-bound on the packed-weight read (informational); prefill (large M) is gated on
                      cuBLAS-equivalence (OURS only -- the qutlass column is informational, never gated).
  * bench_hessian  -- accumulate_hessian (full + lower-only) vs a measured fp16 cuBLAS GEMM at the same
                      (N,N,K), plus the torch fp32 baddbmm baseline for reference.

OUTPUT: a formatted stdout summary is always printed. With --json PATH a machine-readable artifact (GPU name,
measured caps, every row, verdicts) is also written, so two machines' runs can be diffed with `compare`.

The qutlass (in-repo CUDA) head-to-heads auto-run when qutlass is importable, folded into the tables above as
informational columns: the fused QUANT comparison in bench_dram, the fp4 MATMUL comparison in bench_fp4.

Run as a module from the source root (so `triton_kernels` is importable):
  CUDA_VISIBLE_DEVICES=<gpu> python -m triton_kernels.bench                      # full gate -> stdout summary
  CUDA_VISIBLE_DEVICES=<gpu> python -m triton_kernels.bench --lenient             # don't gate on single-pass DRAM
  CUDA_VISIBLE_DEVICES=<gpu> python -m triton_kernels.bench --json a6000.json     # + write JSON artifact
  python -m triton_kernels.bench compare a6000.json l40s.json                     # diff two runs (no GPU)
"""
import argparse
import json
import os
import sys

import torch
import triton  # noqa: F401  (kernels autotune on import-time configs)

from triton_kernels import e2m1, xxfp4
from triton_kernels.transform import block_transform
from triton_kernels.fused import transform_fakequant, transform_quantize_pack
from triton_kernels.fp4mm import to_blocked, fp4_matmul_auto, fp4_matmul_tma, pad_k_to_tile, fp4_blockscale_native
from triton_kernels.accumulate_hessian import accumulate_hessian, accumulate_hessian_baseline

# qutlass (the CUDA fp4 kernels) is OPTIONAL: when importable, bench_fp4 also times the qutlass fp4 matmul next
# to ours + bf16. The gate stays qutlass-free (the qutlass column is purely informational).
try:
    import qutlass
    _HAS_QUTLASS: bool = True
except Exception:  # noqa: BLE001 -- qutlass extension absent -> the qutlass column shows '--'
    _HAS_QUTLASS = False

DEV = torch.device('cuda')
POOL_BYTES = 512 << 20     # >> any GPU's L2 so single-pass reads are cold
ITERS, WARMUP = 80, 20     # single-pass cold timing (matmul/Hessian use triton.testing.do_bench)
DRAM_BOUND_FRAC = 0.80     # single-pass kernel is DRAM-bound if >= this fraction of the copy ceiling (diagnostic)
FP4_PREFILL_FRAC = 0.60    # Blackwell native FP4 MMA: prefill should ~match/beat the equivalent bf16 cuBLAS GEMM, gated PER-SHAPE
# Pre-Blackwell (no native FP4 tensor cores) fp4_matmul_auto's prefill falls back to dequant->cuBLAS: a FULL bf16
# GEMM PLUS a dequant of both operands, so it is structurally bounded BELOW bf16, and small-N can't amortize the
# dequant -- worse the faster bf16 is (L40S small-N ~0.47x, but H100's elite bf16 GEMM drives small-N to ~0.23x).
# So a per-shape floor can't span the fleet; instead gate the GEOMEAN over the prefill shapes (matches the summary's
# reported geomean). 0.35 clears L40S 0.59 / A6000 0.68 / H100 0.41 with headroom, while failing a gross regression.
FP4_PREFILL_FRAC_DEQUANT = 0.35  # pre-Blackwell dequant->cuBLAS prefill floor, applied to the GEOMEAN of prefill shapes (gated)
HESS_FRAC = 0.60           # accumulate_hessian(full) must reach >= this fraction of the fp16 cuBLAS GEMM (gated)

MX = dict(group_size=32, scale_dtype=torch.float8_e8m0fnu, scale_scale=4., global_scale=1. / 3.)
NV = dict(group_size=16, scale_dtype=torch.float8_e4m3fn, scale_scale=6., global_scale=.1)
# fused.transform_* takes the quant group via quant_group_size and the transform block from the transform
# shape (coupled here: 32x32 / 16x16 -> GT == GQ), so it does not take group_size.
MX_FUSED = {k: v for k, v in MX.items() if k != 'group_size'}
NV_FUSED = {k: v for k, v in NV.items() if k != 'group_size'}

DRAM_SHAPES = [(16384, 4096), (8192, 8192), (16384, 8192), (32768, 8192)]  # 128..512 MB, all >> L2 -> cold
FP4_SHAPES = [(4096, 4096), (4096, 12288), (8192, 21760)]                  # (K, N): square / gate / 70B up-proj
FP4_BATCHES = [(16, 'decode'), (1024, 'prefill')]                          # one memory-bound M, one compute-bound M
HESS_SIZES = [4096, 8192, 16384]                                          # square N = K (hidden = batch)


# --------------------------------------------------------------------------------------------------------
# measured ceilings + the shared roofline check
# --------------------------------------------------------------------------------------------------------
def _pool(M, K):
    nbuf = max(3, -(-POOL_BYTES // (M * K * 2)))  # ceil; bf16 = 2 bytes
    return [torch.randn(M, K, dtype=torch.bfloat16, device=DEV) for _ in range(nbuf)]


def bench_cold_ms(run_fn, pool):
    n = len(pool)
    for i in range(WARMUP):
        run_fn(pool[i % n])
    torch.cuda.synchronize()
    s = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    e = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    for i in range(ITERS):
        s[i].record()
        run_fn(pool[i % n])
        e[i].record()
    torch.cuda.synchronize()
    t = sorted(a.elapsed_time(b) for a, b in zip(s, e))
    return t[len(t) // 2]  # median ms


def copy_ceiling() -> float:
    """Measured cold read+write DRAM-copy bandwidth (GB/s) -- the bandwidth roofline for this GPU."""
    N = 64 << 20  # 64M bf16 = 128 MB
    src = [torch.randn(N, dtype=torch.bfloat16, device=DEV) for _ in range(4)]
    dst = torch.empty(N, dtype=torch.bfloat16, device=DEV)
    ms: float = bench_cold_ms(lambda x: dst.copy_(x), src)
    del src, dst
    torch.cuda.empty_cache()
    return (2 * N * 2) / (ms * 1e-3) / 1e9  # read src + write dst


def matmul_ceiling(dtype: torch.dtype, n: int = 8192) -> float:
    """Measured peak cuBLAS GEMM throughput (TF/s) for a square n x n matmul -- the compute roofline anchor."""
    a = torch.randn(n, n, dtype=dtype, device=DEV)
    b = torch.randn(n, n, dtype=dtype, device=DEV)
    ms: float = triton.testing.do_bench(lambda: a @ b)
    del a, b
    torch.cuda.empty_cache()
    return 2. * n ** 3 / (ms * 1e-3) / 1e12


def _gemm_ms(m: int, n: int, k: int, dtype: torch.dtype) -> float:
    """Time (ms) of the equivalent cuBLAS GEMM a(m,k) @ b(n,k).T -- the per-shape compute floor for FP4 matmul."""
    a = torch.randn(m, k, dtype=dtype, device=DEV)
    b = torch.randn(n, k, dtype=dtype, device=DEV)
    ms: float = triton.testing.do_bench(lambda: a @ b.transpose(-2, -1))
    del a, b
    torch.cuda.empty_cache()
    return ms


def _hessian_gemm_ms(n: int, dtype: torch.dtype = torch.float16) -> float:
    """Time (ms) of the equivalent cuBLAS GEMM a.T @ a (a:(n,n)) -- the compute floor for accumulate_hessian."""
    a = torch.randn(n, n, dtype=dtype, device=DEV)
    ms: float = triton.testing.do_bench(lambda: a.transpose(0, 1) @ a)
    del a
    torch.cuda.empty_cache()
    return ms


def _fp4_bytes(m: int, n: int, k: int, group_size: int, out_itemsize: int = 2) -> int:
    """Logical bytes an ideal FP4 matmul moves: read packed A+B (0.5 B/elem) + scales (1 B) + write C."""
    packed: int = (m * k + n * k) // 2
    scales: int = (m * k + n * k) // group_size
    return packed + scales + m * n * out_itemsize


def _roofline(actual_ms: float, n_bytes: int, ceiling_gbps: float, frac: float,
              cublas_ms: float = 0.) -> tuple[float, str, bool]:
    """
    Achievable lower-bound time = max(DRAM read+write @ copy ceiling, equivalent cuBLAS GEMM); the kernel's
    efficiency = floor / actual. `eff > 1` is valid (kernel beats the bf16 GEMM -- Blackwell native FP4, or
    the Hessian symmetry win) and is returned uncapped. Returns (efficiency, binding resource, ok).
    """
    mem_ms: float = n_bytes / (ceiling_gbps * 1e9) * 1e3
    floor_ms: float = max(mem_ms, cublas_ms)
    binding: str = 'GEMM' if cublas_ms > mem_ms else 'MEM'
    eff: float = floor_ms / actual_ms
    return eff, binding, eff >= frac


def _gmean(xs: list[float]) -> float:
    return torch.as_tensor(xs).log().mean().exp().item() if xs else float('nan')


# --------------------------------------------------------------------------------------------------------
# section 1 -- single-pass kernels vs the DRAM copy ceiling
# --------------------------------------------------------------------------------------------------------
def _qut_match(our_fn, qut_fn, x: torch.Tensor) -> float | None:
    """
    Quant bit-agreement of our transform_quantize_pack vs the qutlass fused quant on the same x: the min fraction
    of byte-identical packed-e2m1 codes and (sliced) per-group scale bytes (1.0 = byte-identical). None if the
    qutlass kernel rejects the shape.
    """
    try:
        po, so = our_fn(x)
        qe, qsc = qut_fn(x)
        m, c = so.shape[-2], so.shape[-1]
        code: float = po.view(dtype=torch.uint8).eq(qe.view(dtype=torch.uint8)).float().mean().item()
        scale: float = so.view(dtype=torch.uint8).eq(qsc[:m, :c].reshape(m, c).view(dtype=torch.uint8)).float().mean().item()
        return min(code, scale)
    except Exception:  # noqa: BLE001 -- qutlass shape reject / dtype mismatch -> no match value
        return None


def bench_dram(ceiling: float) -> tuple[list[dict], bool]:
    thresh: float = DRAM_BOUND_FRAC * ceiling
    print(f"== single-pass kernels vs DRAM copy ceiling (bound = >= {DRAM_BOUND_FRAC:.0%} of {ceiling:.0f} GB/s = {thresh:.0f} GB/s) ==")
    print("# qut = qutlass fused transform+quant+pack on the same x (informational; the transposed transform is precomputed offline,"
          " so only the kernel is timed). match = bit-agreement (1.00 = byte-identical codes+scales)."
          if _HAS_QUTLASS else "# qutlass not importable -> the 'qut'/'speedup'/'match' columns show '--'.")
    hdr = (f"{'file':>12} {'op':>36} {'M':>6} {'K':>6} | {'time(us)':>9} | {'ours GB/s':>9} | "
           f"{'qut GB/s':>9} | {'speedup':>8} | {'match':>6} | {'bound':<15}")  # pad to the widest bound value ('compute (info)' / '✗ MEM <80%') so the rule spans the full table
    print(hdr)
    print('-' * len(hdr))
    rows: list[dict] = []

    for (M, K) in DRAM_SHAPES:
        x_pool = _pool(M, K)
        MK = M * K
        rd, wr, pk = MK * 2, MK * 2, MK // 2  # read x (bf16), write y (bf16), packed fp4 (0.5 B/elem)
        sc_mx, sc_nv = MK // MX['group_size'], MK // NV['group_size']  # 1-byte scales per group

        def run(file, op, pool, fn, nbytes, qut_fn=None, match=None, gated=True, streaming=True):
            ms: float = bench_cold_ms(fn, pool)
            bw: float = nbytes / (ms * 1e-3) / 1e9
            eff, _, ok = _roofline(ms, nbytes, ceiling, DRAM_BOUND_FRAC)
            # A transform-bearing op (streaming=False) that misses the copy-bandwidth bar is COMPUTE-bound on this
            # GPU -- the per-element transform matmul outruns the memory pipeline on low-FLOP parts. That is optimal,
            # not a regression, so demote it to informational instead of failing the gate. Pure-streaming kernels
            # (load/store only, ~0 arithmetic intensity) stay strictly gated: a sub-80% there IS a memory regression.
            compute_bound: bool = gated and not streaming and not ok
            row_gated: bool = gated and not compute_bound
            qut_bw = speedup = None
            if qut_fn is not None:
                try:  # the qutlass column is informational -> a shape it rejects must not crash the gate
                    qms: float = bench_cold_ms(qut_fn, pool)  # transposed transform is precomputed -> only the kernel is timed
                    qut_bw, speedup = nbytes / (qms * 1e-3) / 1e9, qms / ms  # speedup > 1 => ours faster
                except Exception:  # noqa: BLE001
                    qut_bw = speedup = None
            rows.append({'file': file, 'op': op, 'M': M, 'K': K, 'time_us': ms * 1e3, 'gbps': bw, 'qut_gbps': qut_bw,
                         'speedup': speedup, 'match': match, 'frac': eff, 'pass': ok, 'gated': row_gated,
                         'compute_bound': compute_bound})
            qut_s: str = f'{qut_bw:.0f}' if qut_bw is not None else '--'
            sp_s: str = f'{speedup:.2f}x' if speedup is not None else '--'
            mt_s: str = f'{match:.2f}' if match is not None else '--'
            bd_s: str = ('compute (info)' if compute_bound else ('✓' if ok else '✗ MEM <80%')) if gated else 'info'
            print(f"{file:>12} {op:>36} {M:>6} {K:>6} | {ms*1e3:9.1f} | {bw:9.0f} | {qut_s:>9} | {sp_s:>8} | {mt_s:>6} | {bd_s}")

        # transform.py
        H3 = torch.randn(K // 32, 32, 32, dtype=torch.bfloat16, device=DEV) * 32 ** -.5  # per-block transform, G=32
        run('transform.py', 'block_transform', x_pool, lambda x: block_transform(x, H3), rd + wr, streaming=False)

        # xxfp4.py (mxfp4 e8m0, + one nvfp4 e4m3 fake_quant to exercise the to_e4m3 path). These compute a per-group
        # absmax->scale and/or pack 4-bit codes -> real per-element ALU, so on a very-high-bandwidth GPU (H100 HBM3
        # ~2.85 TB/s) they become ALU-bound BELOW the copy ceiling -- optimal, not memory-regressed -> streaming=False.
        run('xxfp4.py', 'fake_quantize[mx]', x_pool, lambda x: xxfp4.fake_quantize(x, **MX), rd + wr, streaming=False)
        run('xxfp4.py', 'quantize_pack[mx]', x_pool, lambda x: xxfp4.quantize_pack(x, **MX), rd + pk + sc_mx, streaming=False)
        xx_pk = [xxfp4.quantize_pack(xb, **MX) for xb in x_pool]
        run('xxfp4.py', 'dequantize[mx]', xx_pk, lambda ps: xxfp4.dequantize(ps[0], ps[1], global_scale=MX['global_scale']), pk + sc_mx + wr, streaming=False)
        run('xxfp4.py', 'fake_quantize[nv]', x_pool, lambda x: xxfp4.fake_quantize(x, **NV), rd + wr, streaming=False)

        # e2m1.py (grid-only; no per-group scale). fake_quantize (round only) and dequantize (unpack+LUT) are pure
        # streaming and saturate even H100's HBM3 -> stay strictly gated. quantize_pack adds 4-bit packing ALU -> streaming=False.
        run('e2m1.py', 'fake_quantize', x_pool, lambda x: e2m1.fake_quantize(x), rd + wr)
        run('e2m1.py', 'quantize_pack', x_pool, lambda x: e2m1.quantize_pack(x), rd + pk, streaming=False)
        e_pk = [e2m1.quantize_pack(xb) for xb in x_pool]
        run('e2m1.py', 'dequantize', e_pk, lambda p: e2m1.dequantize(p), pk + wr)

        # fused.py -- transform + fp4 quant. The transform_quantize_pack rows also race the qutlass fused quant
        # (fusedQuantize{Mx,WushMx,Nv}): qutlass computes x@b, we compute x@bᵀ, so its transform `b` is the
        # precomputed TRANSPOSE (a default arg => evaluated once here, OFFLINE, not inside the timed kernel).
        # GT=128 is occupancy-bound for us -> informational (not gated) so it can't break the DRAM verdict.
        H2_mx = torch.randn(32, 32, dtype=torch.bfloat16, device=DEV) * 32 ** -.5
        H2_nv = torch.randn(16, 16, dtype=torch.bfloat16, device=DEV) * 16 ** -.5
        run('fused.py', 'transform_fakequant[mx]', x_pool, lambda x: transform_fakequant(x, H2_mx, **MX_FUSED), rd + wr, streaming=False)
        run('fused.py', 'transform_fakequant[nv]', x_pool, lambda x: transform_fakequant(x, H2_nv, **NV_FUSED), rd + wr, streaming=False)
        if _HAS_QUTLASS:
            gsc = torch.as_tensor([1. / NV['global_scale']], dtype=torch.float32, device=DEV)  # qutlass nvfp4 global scale (old convention -> feed 1/gs); (1,)-shaped (qutlass binding asserts dim==1 && size(0)==1)
            Wp = torch.randn(K // 32, 32, 32, dtype=torch.bfloat16, device=DEV) * 32 ** -.5     # per-block (Wush)
            Wg64 = torch.randn(64, 64, dtype=torch.bfloat16, device=DEV) * 64 ** -.5            # decoupled GT=64
            Wg128 = torch.randn(128, 128, dtype=torch.bfloat16, device=DEV) * 128 ** -.5        # decoupled GT=128
            # (op, ours_fn, qutlass_fn (transposed transform precomputed in the default arg), quant group, gated)
            variants = [  # op tag = the transform's shape (CxGxG per-block / GTxGT shared); quant group stays 32/16
                ('transform_quantize_pack[mx] 32x32', lambda x: transform_quantize_pack(x, H2_mx, **MX_FUSED),
                 lambda x, b=H2_mx.T.contiguous(): qutlass.fusedQuantizeMx(x, b, method='abs_max'), 32, True),
                ('transform_quantize_pack[mx] Cx32x32', lambda x: transform_quantize_pack(x, Wp, **MX_FUSED),
                 lambda x, b=Wp.view(-1, 32).T.contiguous(): qutlass.fusedQuantizeWushMx(x, b), 32, False),
                ('transform_quantize_pack[mx] 64x64', lambda x: transform_quantize_pack(x, Wg64, quant_group_size=32, **MX_FUSED),
                 lambda x, b=Wg64.T.contiguous(): qutlass.fusedQuantizeMx(x, b, method='abs_max'), 32, False),
                ('transform_quantize_pack[mx] 128x128', lambda x: transform_quantize_pack(x, Wg128, quant_group_size=32, **MX_FUSED),
                 lambda x, b=Wg128.T.contiguous(): qutlass.fusedQuantizeMx(x, b, method='abs_max'), 32, False),
                ('transform_quantize_pack[nv] 16x16', lambda x: transform_quantize_pack(x, H2_nv, **NV_FUSED),
                 lambda x, b=H2_nv.T.contiguous(): qutlass.fusedQuantizeNv(x, b, gsc), 16, False),
            ]
            for op, our_fn, qut_fn, grp, gated in variants:
                run('fused.py', op, x_pool, our_fn, rd + pk + (sc_mx if grp == 32 else sc_nv),
                    qut_fn=qut_fn, match=_qut_match(our_fn, qut_fn, x_pool[0]), gated=gated, streaming=False)
        else:
            run('fused.py', 'transform_quantize_pack[mx] 32x32', x_pool, lambda x: transform_quantize_pack(x, H2_mx, **MX_FUSED), rd + pk + sc_mx, streaming=False)

        del x_pool, xx_pk, e_pk
        torch.cuda.empty_cache()
        print()

    gated_rows = [r for r in rows if r['gated']]
    n_bound: int = sum(r['pass'] for r in gated_rows)
    ok: bool = n_bound == len(gated_rows)
    print(f"single-pass: {n_bound}/{len(gated_rows)} DRAM-bound (gated; qutlass-comparison rows are informational)  {'PASS' if ok else 'FAIL'}\n")
    return rows, ok


# --------------------------------------------------------------------------------------------------------
# section 2 -- FP4 matmul (production router) vs a measured bf16 cuBLAS GEMM at the same shape
# --------------------------------------------------------------------------------------------------------
def bench_fp4(ceiling: float) -> tuple[list[dict], bool]:
    blackwell: bool = fp4_blockscale_native(DEV.index if DEV.index is not None else torch.cuda.current_device())
    # Arch-aware prefill gate: native FP4 (Blackwell) is held to "~match/beat bf16"; the pre-Blackwell
    # dequant->cuBLAS fallback is held to the looser "approach bf16" floor (it can't beat a full bf16 GEMM).
    prefill_frac: float = FP4_PREFILL_FRAC if blackwell else FP4_PREFILL_FRAC_DEQUANT
    print("== FP4 matmul: our triton fp4_matmul_auto vs qutlass CUDA vs bf16 cuBLAS at the same shape ==")
    print("# decode (small M): DRAM-bound on the packed-weight read -- informational (not gated).")
    print("# prefill timing hoists the to_blocked scale swizzle out of the timed region (one-time operand prep,"
          " like quantize_pack; qutlass + bf16 are timed the same way) and times the prefill kernel directly.")
    if blackwell:
        print(f"# prefill: Blackwell native FP4 MMA -- FP4 should BEAT bf16 (ours/bf16 > 1, eff > 1 = win). gate ours >= {prefill_frac:.0%}.")
    else:
        print(f"# prefill: pre-Blackwell dequant->cuBLAS -- FP4 should APPROACH bf16. gate GEOMEAN eff >= {prefill_frac:.0%} (per-shape eff shown; small-N is structurally dequant-bound).")
    print("# qut = qutlass CUDA fp4 matmul on the SAME operands (informational, NOT gated); 'match' = rel_l2(ours, qut)."
          if _HAS_QUTLASS else "# qutlass not importable -> the 'qut' columns show '--' (the gate is qutlass-free).")
    hdr = (f"{'fmt':>6} {'regime':>8} {'MxNxK':>20} | {'ours TF/s':>9} | {'qut TF/s':>9} | {'bf16 TF/s':>9} | "
           f"{'ours/bf16':>9} | {'qut/bf16':>8} | {'ours/qut':>8} | {'bound':>5} | {'eff':>6} | {'status':>6} | {'match':>6}")
    print(hdr)
    print('-' * len(hdr))
    rows: list[dict] = []

    for fmt, cfg in (('mxfp4', MX), ('nvfp4', NV)):
        alpha: float = cfg['global_scale'] ** 2.
        gs: int = cfg['group_size']
        qmm = (qutlass.matmul_mxf4_bf16_tn if fmt == 'mxfp4' else qutlass.matmul_nvf4_bf16_tn) if _HAS_QUTLASS else None
        for (K, N) in FP4_SHAPES:
            for M, regime in FP4_BATCHES:
                xa = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
                xb = torch.randn(N, K, dtype=torch.bfloat16, device=DEV)
                ap, asc = xxfp4.quantize_pack(xa, **cfg)  # real packed codes + normal-range scales (row-major)
                bp, bsc = xxfp4.quantize_pack(xb, **cfg)
                # The to_blocked scale swizzle is one-time operand PREP -- like the quantize_pack above, and like
                # how the qutlass column (pre-swizzled, line ~350) and the bf16 cuBLAS floor are timed. So for the
                # gated PREFILL throughput, hoist it OUT of the timed region and time our prefill kernel directly,
                # apples-to-apples with qutlass / bf16 (in real serving the weight scale is swizzled once at load;
                # only the activation swizzle is per-call, and -- like activation quant -- it is prep, not matmul).
                if regime == 'prefill' and blackwell:
                    app, bpp, ascr, bscr = pad_k_to_tile(ap, bp, asc, bsc, gs)  # K -> multiple of 128 (no-op for the 128-aligned bench shapes); mirrors fp4_matmul_auto's prefill path
                    asw, bsw = to_blocked(ascr), to_blocked(bscr)
                    fp4_ms: float = triton.testing.do_bench(lambda: fp4_matmul_tma(app, bpp, asw, bsw, alpha), warmup=10, rep=30)
                else:  # decode (informational, not gated) / pre-Blackwell: the pointer / dequant path takes row-major scales (no swizzle)
                    fp4_ms = triton.testing.do_bench(lambda: fp4_matmul_auto(ap, bp, asc, bsc, alpha), warmup=10, rep=30)
                bf16_ms: float = _gemm_ms(M, N, K, torch.bfloat16)
                flops: float = 2. * M * N * K
                fp4_tf, bf16_tf = flops / (fp4_ms * 1e-3) / 1e12, flops / (bf16_ms * 1e-3) / 1e12

                # qutlass fp4 matmul on the SAME packed operands: uint8 codes + blocked scales (flat mxfp4 /
                # (-1, K//16) nvfp4) + an fp32 alpha tensor. Informational; a rejected shape -> '--'.
                qut_tf = match = None
                if qmm is not None:
                    ap_u8, bp_u8 = ap.view(dtype=torch.uint8), bp.view(dtype=torch.uint8)
                    asb, bsb = to_blocked(asc), to_blocked(bsc)
                    if fmt == 'nvfp4':
                        asb, bsb = asb.view(-1, K // 16), bsb.view(-1, K // 16)
                    alpha_t: torch.Tensor = torch.as_tensor(alpha, dtype=torch.float32, device=DEV)  # 0-d (qutlass reads data_ptr)
                    try:
                        ours_out: torch.Tensor = fp4_matmul_auto(ap, bp, asc, bsc, alpha)
                        qut_out: torch.Tensor = qmm(ap_u8, bp_u8, asb, bsb, alpha_t)
                        match = ((ours_out.to(torch.float32) - qut_out.to(torch.float32)).norm() / qut_out.to(torch.float32).norm()).item()
                        qut_ms: float = triton.testing.do_bench(lambda: qmm(ap_u8, bp_u8, asb, bsb, alpha_t), warmup=10, rep=30)
                        qut_tf = flops / (qut_ms * 1e-3) / 1e12
                    except Exception:  # noqa: BLE001 -- a shape qutlass rejects -> n/a, keep the sweep alive
                        qut_tf = match = None

                eff, binding, ok = _roofline(fp4_ms, _fp4_bytes(M, N, K, gs), ceiling, prefill_frac, cublas_ms=bf16_ms)
                gated: bool = regime == 'prefill'
                if not gated:
                    status = 'info'
                elif blackwell:
                    status = 'PASS' if ok else 'FAIL'    # native FP4 gated per-shape
                else:
                    status = 'PASS' if ok else 'low'     # pre-Blackwell gates on the geomean (below), not per-shape
                qut_bf16: float | None = (qut_tf / bf16_tf) if qut_tf is not None else None
                ours_qut: float | None = (fp4_tf / qut_tf) if qut_tf is not None else None
                rows.append({'fmt': fmt, 'regime': regime, 'M': M, 'N': N, 'K': K, 'fp4_tflops': fp4_tf,
                             'qut_tflops': qut_tf, 'bf16_tflops': bf16_tf, 'ratio': fp4_tf / bf16_tf,
                             'qut_bf16_ratio': qut_bf16, 'ours_qut_ratio': ours_qut, 'match': match,
                             'binding': binding, 'eff': eff, 'gated': gated, 'pass': ok, 'blackwell': blackwell})
                qut_s: str = f"{qut_tf:.0f}" if qut_tf is not None else '--'
                qb_s: str = f"{qut_bf16:.2f}x" if qut_bf16 is not None else '--'
                oq_s: str = f"{ours_qut:.2f}x" if ours_qut is not None else '--'
                match_s: str = '✓' if (match is not None and match < 2e-2) else (f'⚠{match:.0e}' if match is not None else '--')
                print(f"{fmt:>6} {regime:>8} {f'{M}x{N}x{K}':>20} | {fp4_tf:9.0f} | {qut_s:>9} | {bf16_tf:9.0f} | "
                      f"{fp4_tf/bf16_tf:8.2f}x | {qb_s:>8} | {oq_s:>8} | {binding:>5} | {eff:5.2f}x | {status:>6} | {match_s:>6}")
                del xa, xb, ap, bp, asc, bsc
                torch.cuda.empty_cache()

    gated_effs = [r['eff'] for r in rows if r['gated']]
    # Blackwell: native FP4 must clear the floor on EVERY prefill shape. Pre-Blackwell dequant fallback: gate the
    # GEOMEAN (small-N is structurally dequant-overhead-bound, esp. against H100's elite bf16 -> per-shape is too strict).
    ok = all(e >= prefill_frac for e in gated_effs) if blackwell else (_gmean(gated_effs) >= prefill_frac)
    print(f"FP4 prefill cuBLAS-equivalence: {'PASS' if ok else 'FAIL'}\n")
    return rows, ok


# --------------------------------------------------------------------------------------------------------
# section 3 -- accumulate_hessian vs a measured fp16 cuBLAS GEMM at the same shape
# --------------------------------------------------------------------------------------------------------
def bench_hessian(ceiling: float) -> tuple[list[dict], bool]:
    print("== accumulate_hessian (X^T X, fp32 accum) vs fp16 cuBLAS at the same shape ==")
    print(f"# full does the same FLOPs as the GEMM -> gated at eff >= {HESS_FRAC:.0%}. lower-only exploits Hessian")
    print("# symmetry (~half the work, eff can exceed 1). torch = fp32 baddbmm baseline (true fp32, not TF32).")
    hdr = f"{'N=K':>6} | {'full TF/s':>9} | {'lower TF/s':>10} | {'torch TF/s':>10} | {'fp16 GEMM':>9} | {'full/cap':>8} | {'lower/torch':>11} | status"
    print(hdr)
    print('-' * len(hdr))
    rows: list[dict] = []

    for n in HESS_SIZES:
        a = torch.randn(n, n, device=DEV, dtype=torch.float16)   # (size_batch, size_hidden)
        c = torch.randn(n, n, device=DEV, dtype=torch.float32)   # Hessian (hidden, hidden), fp32 accumulator
        flops: float = 2. * n ** 3
        tf = lambda ms: flops / (ms * 1e-3) / 1e12
        full_ms: float = triton.testing.do_bench(lambda: accumulate_hessian(c, a, save_lower_only=False, compute_lower_only=False))
        low_ms: float = triton.testing.do_bench(lambda: accumulate_hessian(c, a, save_lower_only=False, compute_lower_only=True))
        torch_ms: float = triton.testing.do_bench(lambda: accumulate_hessian_baseline(c, a))
        gemm_ms: float = _hessian_gemm_ms(n, torch.float16)
        eff, _, ok = _roofline(full_ms, 0, ceiling, HESS_FRAC, cublas_ms=gemm_ms)  # compute-bound: GEMM floor
        rows.append({'N': n, 'K': n, 'full_tflops': tf(full_ms), 'lower_tflops': tf(low_ms),
                     'torch_tflops': tf(torch_ms), 'gemm_tflops': tf(gemm_ms), 'eff': eff, 'pass': ok})
        print(f"{n:>6} | {tf(full_ms):8.0f}  | {tf(low_ms):9.0f}  | {tf(torch_ms):9.0f}  | {tf(gemm_ms):8.0f}  | {eff:7.2f}x | {torch_ms/low_ms:10.2f}x | {'PASS' if ok else 'FAIL'}")
        del a, c
        torch.cuda.empty_cache()

    ok = all(r['pass'] for r in rows)
    print(f"Hessian fp16-cuBLAS-equivalence: {'PASS' if ok else 'FAIL'}\n")
    return rows, ok


# --------------------------------------------------------------------------------------------------------
# summary + JSON artifact + cross-machine compare
# --------------------------------------------------------------------------------------------------------
def print_summary(caps: dict, dram: tuple, fp4: tuple, hess: tuple, gate_dram: bool = True) -> bool:
    dram_rows, dram_ok = dram
    fp4_rows, fp4_ok = fp4
    hess_rows, hess_ok = hess
    overall: bool = (dram_ok or not gate_dram) and fp4_ok and hess_ok
    gated_dram = [r for r in dram_rows if r.get('gated', True)]  # qutlass-comparison + demoted compute-bound rows are informational
    n_bound: int = sum(r['pass'] for r in gated_dram)
    not_bound: list[str] = sorted({r['op'] for r in gated_dram if not r['pass']})                  # genuine sub-80% memory regressions
    compute_bound_ops: list[str] = sorted({r['op'] for r in dram_rows if r.get('compute_bound')})  # transform ops demoted to informational on this GPU
    fp4_prefill = [r['ratio'] for r in fp4_rows if r['regime'] == 'prefill']
    hess_eff = [r['eff'] for r in hess_rows]
    dram_tag: str = (('PASS' if dram_ok else 'FAIL') + '  [gated]') if gate_dram else 'diagnostic, not gated'

    bar = '=' * 64
    print(bar)
    print(f"SUMMARY  [{caps['gpu']}]")
    print(f"  measured caps: copy {caps['copy_gbps']:.0f} GB/s | "
          f"bf16 GEMM {caps['gemm_tflops']['bf16']:.0f} TF/s | fp16 GEMM {caps['gemm_tflops']['fp16']:.0f} TF/s")
    # build each section's content string, pad them all to a common width so the trailing PASS/[gated]/[info] tags line up
    qut_bf16 = [r['qut_bf16_ratio'] for r in fp4_rows if r['regime'] == 'prefill' and r.get('qut_bf16_ratio') is not None]
    ours_qut = [r['ours_qut_ratio'] for r in fp4_rows if r['regime'] == 'prefill' and r.get('ours_qut_ratio') is not None]
    dram_c: str = f"{n_bound}/{len(gated_dram)} bound (>= {DRAM_BOUND_FRAC:.0%} of ceiling)"
    fp4_c: str = f"prefill {_gmean(fp4_prefill):.2f}x bf16 cuBLAS (geomean)"
    hess_c: str = f"{_gmean(hess_eff):.2f}x fp16 cuBLAS (geomean)"
    qut_c: str | None = (f"prefill qutlass {_gmean(qut_bf16):.2f}x bf16 cuBLAS | "
                         f"ours {_gmean(ours_qut):.2f}x qutlass (geomean)") if qut_bf16 else None
    cw: int = max(len(c) for c in (dram_c, fp4_c, hess_c) + ((qut_c,) if qut_c else ()))
    print(f"  {'single-pass DRAM':<16} : {dram_c:<{cw}}   {dram_tag}")
    if not_bound:
        print(f"                     below 80% of copy ceiling (memory regression?): {', '.join(not_bound)}")
    if compute_bound_ops:
        print(f"                     compute-bound on this GPU (informational, not gated): {', '.join(compute_bound_ops)}")
    print(f"  {'FP4 matmul':<16} : {fp4_c:<{cw}}   {'PASS' if fp4_ok else 'FAIL'}  [gated]")
    if qut_c:
        print(f"  {'FP4 vs qutlass':<16} : {qut_c:<{cw}}   [info]")
    print(f"  {'Hessian (full)':<16} : {hess_c:<{cw}}   {'PASS' if hess_ok else 'FAIL'}  [gated]")
    if gate_dram and not dram_ok:
        print("  note: a PURE-STREAMING kernel fell below 80% of the copy ceiling -- unlike a compute-bound transform")
        print("        op (auto-demoted to informational above), this usually signals a real memory regression. Use")
        print("        `compare` vs a reference machine to confirm, or --lenient to demote the single-pass gate.")
    print('-' * len(bar))
    print(f"  OVERALL: {'PASS' if overall else 'FAIL'}")
    print(bar)
    return overall


def build_results(caps: dict, dram_rows: list, fp4_rows: list, hess_rows: list, verdict: dict) -> dict:
    return {'gpu': caps['gpu'], 'triton': triton.__version__, 'torch': torch.__version__,
            'caps': {'copy_gbps': caps['copy_gbps'], 'gemm_tflops': caps['gemm_tflops']},
            'dram': dram_rows, 'fp4': fp4_rows, 'hessian': hess_rows, 'verdict': verdict}


def write_json(path: str, results: dict) -> None:
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"# wrote {path}")


def compare(path_a: str, path_b: str) -> None:
    """Diff two JSON result files side by side (A = baseline, ratio = B/A). No GPU needed."""
    with open(path_a) as f:
        a = json.load(f)
    with open(path_b) as f:
        b = json.load(f)
    print(f"# compare   A = {a['gpu']}   B = {b['gpu']}")
    ca, cb = a['caps'], b['caps']
    print(f"# caps  copy {ca['copy_gbps']:.0f} vs {cb['copy_gbps']:.0f} GB/s | "
          f"bf16 GEMM {ca['gemm_tflops']['bf16']:.0f} vs {cb['gemm_tflops']['bf16']:.0f} | "
          f"fp16 GEMM {ca['gemm_tflops']['fp16']:.0f} vs {cb['gemm_tflops']['fp16']:.0f} TF/s")

    def table(title: str, rows_a: list, rows_b: list, key, metric: str, unit: str) -> None:
        idx_b = {key(r): r for r in rows_b}
        w: int = max([len(key(r)) for r in rows_a] + [len('op')])  # size the op column to the widest key (labels vary in length)
        hdr = f"  {'op':>{w}} | {'A':>8} | {'B':>8} | {'B/A':>6}"
        print(f"\n-- {title}  ({metric}, {unit}) --")
        print(hdr)
        print('  ' + '-' * (len(hdr) - 2))
        for ra in rows_a:
            k = key(ra)
            rb = idx_b.get(k)
            va: float = ra[metric]
            vb = rb[metric] if rb is not None else None
            ratio: str = f"{vb / va:5.2f}x" if (rb is not None and va) else '   -- '
            vb_s: str = f"{vb:8.0f}" if vb is not None else '     N/A'
            print(f"  {k:>{w}} | {va:8.0f} | {vb_s} | {ratio}")

    table('single-pass DRAM', a['dram'], b['dram'],
          lambda r: f"{r['op']} {r['M']}x{r['K']}", 'gbps', 'GB/s')
    table('FP4 matmul', a['fp4'], b['fp4'],
          lambda r: f"{r['fmt']} {r['regime']} {r['M']}x{r['N']}x{r['K']}", 'fp4_tflops', 'TF/s')
    table('Hessian (full)', a['hessian'], b['hessian'],
          lambda r: f"N={r['N']}", 'full_tflops', 'TF/s')
    print(f"\n# overall: A {a['verdict']['overall']}  |  B {b['verdict']['overall']}")


# --------------------------------------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------------------------------------
def main(json_path: str | None = None, gate_dram: bool = True) -> int:
    print(f"# {torch.cuda.get_device_name(0)} | triton {triton.__version__} | torch {torch.__version__}")
    ceiling: float = copy_ceiling()
    caps: dict = {'gpu': torch.cuda.get_device_name(0), 'copy_gbps': ceiling,
                  'gemm_tflops': {'bf16': matmul_ceiling(torch.bfloat16), 'fp16': matmul_ceiling(torch.float16)}}
    print(f"# measured caps: copy {ceiling:.0f} GB/s | "
          f"bf16 GEMM {caps['gemm_tflops']['bf16']:.0f} TF/s | fp16 GEMM {caps['gemm_tflops']['fp16']:.0f} TF/s")
    print(f"# single-pass cold timing: median over {ITERS} iters, input rotated through a {POOL_BYTES>>20} MB pool (> L2)\n")

    dram = bench_dram(ceiling)  # the first table also races our quant vs the qutlass fused quant (informational columns)
    fp4 = bench_fp4(ceiling)
    hess = bench_hessian(ceiling)
    overall: bool = print_summary(caps, dram, fp4, hess, gate_dram=gate_dram)

    if json_path:
        verdict = {'dram': dram[1], 'fp4': fp4[1], 'hessian': hess[1], 'overall': overall}
        write_json(json_path, build_results(caps, dram[0], fp4[0], hess[0], verdict))
    return 0 if overall else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='python -m triton_kernels.bench',
                                     description='Unified performance gate for the triton_kernels/ kernels.')
    parser.add_argument('mode', nargs='?', default=None, help="'compare' to diff two JSON result files (no GPU)")
    parser.add_argument('files', nargs='*', help='two result JSONs when mode=compare (A B; ratio = B/A)')
    parser.add_argument('--json', dest='json_path', metavar='PATH', default=None,
                        help='also write a structured JSON results artifact to PATH')
    parser.add_argument('--lenient', action='store_true',
                        help='do not gate OVERALL on the single-pass DRAM section (treat it as a diagnostic)')
    args = parser.parse_args()

    if args.mode == 'compare':
        if len(args.files) != 2:
            parser.error('compare needs exactly two JSON files: compare A.json B.json')
        compare(args.files[0], args.files[1])
        sys.exit(0)
    if args.mode is not None:
        parser.error(f"unknown mode '{args.mode}' (only 'compare' is supported)")

    code = main(json_path=args.json_path, gate_dram=not args.lenient)  # main() also auto-runs the qutlass quant head-to-head
    sys.exit(code)
