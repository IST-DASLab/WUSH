# Correctness tests for the fused WUSH + MXFP4-quantization kernel
# (`fusedQuantizeWushMx`) described in Section 6 ("GPU Kernel Support") of the
# WUSH paper. Unlike the Hadamard kernel (`fusedQuantizeMx`), which reuses a
# single (G, G) transform across every block, the WUSH kernel applies a
# *distinct* (G, G) transform to each of the C = K / G blocks of the activation
# row, then absmax-quantizes each 32-element block to MXFP4. The per-block
# transform is selected implicitly via a thread-block offset (see the
# `wush_offset` patch in cutlass/gemm/threadblock/mma_multistage.h).
#
# This is a plain script (no pytest): run it directly.
# It exits 0 if all cases pass, 1 otherwise.

import sys

import numpy as np
import torch

from qutlass import fusedQuantizeWushMx, fusedQuantizeMx

try:
    from scipy.linalg import hadamard
except ImportError:  # scipy is only needed for the Hadamard-equivalence test
    hadamard = None

DTYPE = torch.bfloat16
DEVICE = torch.device("cuda")
GROUP_SIZE = 32  # MXFP4 block size G; also the WUSH transform dimension.


def get_hadamard_matrix(group_size: int, dtype: torch.dtype, device: torch.device):
    return torch.tensor(
        hadamard(group_size) * group_size**-0.5, dtype=dtype, device=device
    )


def _rtne_fp4(x: torch.Tensor):
    """Round-to-nearest-even onto the E2M1 (FP4) grid; returns (values, packed codes)."""
    device = x.device
    grid = torch.tensor(
        [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.0,
         0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=x.dtype,
        device=device,
    )
    grid_int = torch.tensor(
        [-1, -2, -3, -4, -5, -6, -7, -8, 0, 1, 2, 3, 4, 5, 6, 7],
        dtype=torch.uint8,
        device=device,
    )
    inds = torch.bucketize(x, grid)
    lo, hi = (inds - 1).clamp(min=0, max=15), inds.clamp(min=0, max=15)
    g_lo, g_hi = grid[lo], grid[hi]
    pick_hi = (g_hi - x < x - g_lo) | (g_hi - x == x - g_lo) & (grid_int[hi] % 2 == 0)
    y = torch.where(pick_hi, g_hi, g_lo)
    y_int = torch.where(pick_hi, grid_int[hi], grid_int[lo])
    y_int_packed = (y_int[..., 1::2] & 0xF) << 4 | y_int[..., ::2] & 0xF
    return y, y_int_packed


def _dq_fp4(x_e2m1: torch.Tensor, x_e8m0: torch.Tensor, alpha: float):
    """Dequantize packed E2M1 codes with their E8M0 block scales."""
    x_e2m1_i32 = x_e2m1.view(dtype=torch.uint8).to(dtype=torch.int32)
    x_e2m1_unpacked = torch.stack(
        [x_e2m1_i32 & 0xF, (x_e2m1_i32 >> 4) & 0xF], dim=-1
    ).flatten(start_dim=-2)

    grid_dq = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float64,
        device=x_e2m1.device,
    )
    x_fp4_dq = grid_dq[x_e2m1_unpacked]
    scales_dq = x_e8m0.to(torch.float64)
    x_dq = (
        x_fp4_dq.unflatten(dim=-1, sizes=(-1, 32)) * scales_dq[..., None]
    ).flatten(start_dim=-2) / alpha
    return x_dq


def _wush_forward_quantize_ref(x: torch.Tensor, w: torch.Tensor):
    """Reference for the fused WUSH+quant kernel.

    Args:
        x: activations, shape (M, C * G), bf16.
        w: per-block transforms, shape (C, G, G), bf16 (as produced by the
           benchmark's ``get_wush_matrix``). Block ``c`` transforms activation
           block ``c`` as ``x_block @ w[c].T`` (the kernel consumes the matrix
           in transposed/row-major form, see ``B = w.view(-1, G).T`` below).

    Returns:
        (e2m1_ref, e8m0_ref, dq_ref) -- packed FP4 codes, E8M0 block scales, and
        the dequantized activation, all computed in float64.
    """
    m, c, g = x.shape[0], w.shape[0], GROUP_SIZE
    xb = x.reshape(m, c, g).to(torch.float64)
    # x_block @ w[c].T, per block.
    xh = torch.einsum("mcg,chg->mch", xb, w.to(torch.float64)).reshape(m, c * g)

    # Absmax MXFP4: scale = floor(log2(absmax)) as an E8M0 power of two.
    abs_max = xh.unflatten(dim=-1, sizes=(-1, 32)).abs().amax(dim=-1)
    e8m0 = (abs_max + 1e-8).log2().floor().exp2().to(dtype=torch.float8_e8m0fnu)
    scales = e8m0.to(torch.float64)

    # The absmax path scales the normalized values by 3 before rounding
    # (matching `fusedQuantizeMx(method="abs_max")`, dequantized with alpha=3).
    xh_scaled = (
        xh.unflatten(dim=-1, sizes=(-1, 32)) / scales[..., None]
    ).flatten(start_dim=-2) * 3
    _, e2m1 = _rtne_fp4(xh_scaled)
    dq = _dq_fp4(e2m1, e8m0, alpha=3.0)
    return e2m1, e8m0, dq


def _make_wush_input(w: torch.Tensor) -> torch.Tensor:
    """Pack a (C, G, G) per-block transform into the (G, C*G) layout the kernel expects."""
    return w.view(-1, GROUP_SIZE).T.contiguous()


def _seed():
    np.random.seed(0)
    torch.random.manual_seed(0)


@torch.inference_mode()
def check_wush_distinct_block_transforms(m: int, k: int):
    """Each block must use its *own* transform (validates the wush_offset path)."""
    dtype, device, g = DTYPE, DEVICE, GROUP_SIZE
    c = k // g

    x = torch.randn(m, k, dtype=dtype, device=device) * 25.0
    w = torch.randn(c, g, g, dtype=dtype, device=device)

    e2m1, e8m0 = fusedQuantizeWushMx(x, _make_wush_input(w))
    e2m1_ref, _, dq_ref = _wush_forward_quantize_ref(x, w)

    # FP4 codes are layout-independent, so compare them directly. If the
    # per-block offset were broken, every block would read transform 0 and the
    # codes for blocks 1..C-1 would diverge.
    code_mismatch = (e2m1 != e2m1_ref).float().mean()
    assert code_mismatch <= 1e-4, f"FP4 code mismatch too high: {code_mismatch.item():.3e}"

    # Dequantized values (codes + block scales) must match the reference; the
    # tiny tolerance absorbs bf16-vs-float64 rounding at quantization bin edges.
    dq = _dq_fp4(e2m1, e8m0[:m, :c], alpha=3.0)
    torch.testing.assert_close(dq, dq_ref, rtol=0.34, atol=100)
    assert (dq != dq_ref).float().mean() <= 1e-4


@torch.inference_mode()
def check_wush_hadamard_matches_mxfp4_kernel(m: int, k: int):
    """With every block set to the Hadamard transform, the WUSH kernel must
    reproduce the trusted `fusedQuantizeMx(method="abs_max")` kernel bit-for-bit."""
    assert hadamard is not None, "scipy is required for the Hadamard-equivalence test"
    dtype, device, g = DTYPE, DEVICE, GROUP_SIZE
    c = k // g

    x = torch.randn(m, k, dtype=dtype, device=device) * 25.0
    h = get_hadamard_matrix(g, dtype, device)
    w = h.unsqueeze(0).expand(c, g, g).contiguous()

    e2m1_wush, e8m0_wush = fusedQuantizeWushMx(x, _make_wush_input(w))
    e2m1_had, e8m0_had = fusedQuantizeMx(x, h, method="abs_max")

    assert e2m1_wush.equal(e2m1_had), "WUSH-with-Hadamard FP4 codes differ from Hadamard kernel"
    assert e8m0_wush.equal(e8m0_had), "WUSH-with-Hadamard E8M0 scales differ from Hadamard kernel"


# (test_fn, kwargs) cases. Realistic layer shapes: K is a multiple of 128 (so
# C = K/32 is a multiple of 4, matching how the E8M0 scale factors are padded),
# and M covers both tile-aligned and partial-tile batch sizes.
def _build_cases():
    cases = []
    for k in (2048, 4096):
        for m in (16, 257, 1024):
            cases.append((check_wush_distinct_block_transforms, dict(m=m, k=k)))
        for m in (256, 1024):
            cases.append((check_wush_hadamard_matches_mxfp4_kernel, dict(m=m, k=k)))
    return cases


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: CUDA required for these tests.")
        return 0

    cases = _build_cases()
    failures = []
    for fn, kwargs in cases:
        name = fn.__name__ + "[" + ",".join(f"{k}={v}" for k, v in kwargs.items()) + "]"
        _seed()
        try:
            fn(**kwargs)
        except Exception as exc:  # noqa: BLE001 - report any failure per case
            failures.append((name, exc))
            print(f"FAIL {name}: {exc}")
        else:
            print(f"PASS {name}")

    total = len(cases)
    print(f"\n{total - len(failures)}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
