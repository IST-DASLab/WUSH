"""
Block transform (memory-bound block-diagonal matmul).

  y[m, c * GT + j] = sum_i x[m, c * GT + i] * T[c, j, i]     (for each block c: Y_c = X_c @ T_c^T)

x: (..., K), transform: a contiguous (C, GT, GT) tensor with C = K / GT. By default `transform[c]` is the
transform T_c and the kernel reads T_c^T via an index swap (no offline transpose / copy needed).
Pass transposed=True if `transform[c]` already holds T_c^T -- also contiguous, e.g. built via
`T.transpose(-1, -2).contiguous()` (a bare `.transpose(-1, -2)` view is NOT contiguous and is rejected).

REPEATED (shared) transform: pass a 2D (GT, GT) transform to apply the SAME matrix to every block
(y = (x.reshape(M * C, GT) @ T^T).reshape(M, K), e.g. a single Hadamard). Since the kernel reads x only
through strides, block_transform just hands it the flat buffer as (M * C, GT) -- one block per kernel
row, C=1 -- with NO reshape and no copies of the matrix. The only thing the C=1 shape needs is the
right tiling (WIDTH=GT, blocks-per-program scaled by TILE_M instead of WIDTH), which early_config_prune
supplies (see prune_configs); the kernel's NT==1 branch then runs a plain (TILE_M , GT) @ (GT, GT) matmul. Same
result as a (C, GT, GT) tensor with the matrix repeated C times, and just as fast.

Supports:
  - block size GT = any power of two in [1, 128];
  - input dtype = bf16 / fp16 / fp32 (all GT) and fp64 (GT <= 64; fp64 has no tensor-core path
    and its manual working set exceeds resources at GT=128). x and transform share dtype;
  - output dtype is independent (default = input). The accumulator is already fp32 for 16-bit
    inputs, so out_dtype=fp32 keeps full tensor-core precision at no extra cost (and only ~2x
    output write traffic); e.g. block_transform(x_bf16, t_bf16, out_dtype=torch.float32).

Two compute paths, chosen at compile time from GT and the element size:
  - GT >= 16 and not fp64  -> tensor-core tl.dot (fp32 accumulate; ieee precision for fp32);
  - GT < 16  or  fp64      -> manual multiply-accumulate. tl.dot requires the contraction dim K (= GT here)
                             >= 16, so GT < 16 cannot use it; fp64 compiles but has no tensor-core path on
                             Ada (emulated, no faster than the MAC). Accumulates in fp32 (fp64 for fp64 input).

Bandwidth techniques (all paths): a 1D SWIZZLED grid (column-group fast-varying so co-scheduled
programs read contiguous columns), a single WIDE (TILE_M, WIDTH) load/store, and processing
NT = WIDTH//GT consecutive blocks per program. Autotune is over (TILE_M, WIDTH); NT shrinks
as GT grows, keeping the per-program working set bounded.
"""

import torch
import triton
import triton.language as tl

from triton_kernels.guard import device_guard


# Autotune over (TILE_M, WIDTH); WIDTH = contiguous wide-load width in ELEMENTS, and the kernel
# derives NT = WIDTH//GT. WIDTH is counted in elements, not bytes, so NT carries no ISIZE term:
# the smallest WIDTH=128 already clears the per-row vectorization floor for every dtype (256 B at
# bf16, wider for fp32/64), and the one ISIZE-sensitive cost -- per-program working set (bytes) -- is
# left to the autotuner. Keyed on (C, GT, ISIZE, OSIZE), it tunes each block size / dtype-class apart.
TILE_CONFIGS = [
    triton.Config({'TILE_M': 16, 'WIDTH': 128}, num_warps=4, num_stages=2),
    triton.Config({'TILE_M': 32, 'WIDTH': 128}, num_warps=4, num_stages=2),
    triton.Config({'TILE_M': 32, 'WIDTH': 128}, num_warps=8, num_stages=2),
    triton.Config({'TILE_M': 16, 'WIDTH': 256}, num_warps=4, num_stages=2),
    triton.Config({'TILE_M': 32, 'WIDTH': 256}, num_warps=4, num_stages=2),
    triton.Config({'TILE_M': 32, 'WIDTH': 256}, num_warps=8, num_stages=2),
    triton.Config({'TILE_M': 16, 'WIDTH': 512}, num_warps=8, num_stages=2),
    triton.Config({'TILE_M': 32, 'WIDTH': 512}, num_warps=8, num_stages=2),
    # sm_100 (B200, HBM3e ~5.6 TB/s): the transform matmul + wide slab load needs a DEEPER software pipeline
    # (num_stages 3-5) and taller tiles (TILE_M 64/128) to keep enough bytes in flight for the bandwidth.
    # Smaller GPUs keep the entries above.
    triton.Config({'TILE_M': 64, 'WIDTH': 128}, num_warps=2, num_stages=3),
    triton.Config({'TILE_M': 64, 'WIDTH': 128}, num_warps=2, num_stages=4),
    triton.Config({'TILE_M': 64, 'WIDTH': 128}, num_warps=2, num_stages=5),
    triton.Config({'TILE_M': 128, 'WIDTH': 128}, num_warps=4, num_stages=2),
    triton.Config({'TILE_M': 128, 'WIDTH': 128}, num_warps=4, num_stages=4),
    triton.Config({'TILE_M': 64, 'WIDTH': 256}, num_warps=4, num_stages=2),
]


# Shape-aware config pruning (Triton early_config_prune): runs once per new autotune key, BEFORE
# benchmarking, with the kernel's constexprs available in kwargs. The repeated/shared transform is
# dispatched as a C=1 problem (one block wide). There WIDTH>GT would only load masked columns, so
# blocks-per-program must scale via TILE_M, not WIDTH (these are the same lever -- see README). We
# therefore hand C=1 a set of WIDTH=GT configs over a range of TILE_M; computing WIDTH=GT at prune
# time sidesteps the static-config limitation (GT varies) and keeps these configs off every other
# shape's tuning. C>1 (the normal per-block transform) uses the stock (TILE_M, WIDTH) configs.
def prune_configs(configs, named_args: dict = {}, **kwargs):
    a: dict = named_args | kwargs  # kernel args, whether passed positionally (named_args) or by keyword (kwargs)
    C, GT, ISIZE = a['C'], a['GT'], a['ISIZE']
    if C == 1:
        # blocks-per-program is set by TILE_M here (WIDTH=GT), so small GT wants large TILE_M to fill
        # the MMA. The manual MAC path (GT<16 or fp64) materializes a (TILE_M, GT, GT) intermediate, so
        # cap TILE_M there to keep the working set bounded; the tensor-core path can go wide.
        manual: bool = (GT < 16) or (ISIZE == 8)
        tiles: tuple = (16, 32, 64) if manual else (32, 64, 128)
        return [
            triton.Config({'TILE_M': t, 'WIDTH': GT}, num_warps=w, num_stages=2)
            for t in tiles for w in (4, 8)
        ]
    return configs


# load_transform: map program id -> this program's tile (the bandwidth swizzle), load the wide x slab,
# and apply the block transform. fused.py reuses it verbatim and only inserts the quantization before
# storing. It returns the transformed slab plus the tile's two coordinates (rm, c0) ONLY -- the caller
# derives whatever store offsets / masks it needs (mask_m = rm < M, columns = c0*GT + arange(WIDTH)), so
# no offset / mask tensors cross the interface.
#
# Bandwidth swizzle: the 1D grid is folded row-major into an (M-tile) x (column-group) logical grid
# with the column-group fast-varying:  pid = pid_m * N_COL_GROUPS + pid_g,  N_COL_GROUPS = ceil(C/NT).
# One program owns a TILE_M x (NT*GT) tile of Y (M x K = C*GT):
#
#            col-group:  g=0       g=1       g=2       g=3     (NT*GT cols each)
#                       +---------+---------+---------+---------+
#   M-tile  pid_m=0     |  pid 0  |  pid 1  |  pid 2  |  pid 3  |  consecutive pids
#                       +---------+---------+---------+---------+  walk columns of the
#           pid_m=1     |  pid 4  |  pid 5  |  pid 6  |  pid 7  |  SAME rows -> each
#                       +---------+---------+---------+---------+  co-scheduled wave's
#           pid_m=2     |  pid 8  |  pid 9  |  pid 10 |  pid 11 |  loads stay contiguous in DRAM.
#                       +---------+---------+---------+---------+
#
# If pid_m were fast-varying instead, consecutive pids would stride by a whole row of blocks (K
# elements apart) -> scattered loads. Coalescing/vectorization happen per-WARP, never across CTAs, so
# the run that vectorizes is the WIDTH-wide slab loaded WITHIN one program: NT>1 (WIDTH>GT) lengthens
# it past GT (GT=32 bf16: 64 B -> 256 B at WIDTH=128) and amortizes per-CTA overhead over NT blocks.
@triton.jit
def load_transform(
        pid,
        x_ptr, t_ptr,
        M, C, GT: tl.constexpr,
        stride_xm, stride_xk,
        stride_tc, stride_ti, stride_tj,
        TILE_M: tl.constexpr, WIDTH: tl.constexpr,
        NT: tl.constexpr, USE_DOT: tl.constexpr, ACC: tl.constexpr,
):
    """
    pid -> tile (swizzle), load the wide (TILE_M, WIDTH=NT*GT) x slab, apply the block transform.
    Returns (slab, rm, c0): the transformed slab (TILE_M, WIDTH, fp32/ACC) and the tile's two
    coordinates -- rm = the TILE_M row indices it owns, c0 = its first block column (it owns the NT
    blocks c0 .. c0+NT). The caller derives any store offsets / masks from these. The slab holds
    NT blocks of width GT side by side, each y_block = x_block @ T_c^T:
      NT==1: one block (WIDTH==GT) -> plain (TILE_M, GT) @ (GT, GT), no batch / permute (the repeated
                transform with C=1, and the GT=128 per-block case; a batched 1-wide dot is slower).
      NT>1:  NT distinct matrices T_(c0+b) -> batched (NT, TILE_M, GT) @ (NT, GT, GT).
    tl.dot covers 2D and batched 3D; the manual (GT<16 / fp64) path is rank-generic (negative axes).
    """
    N_COL_GROUPS: tl.constexpr = tl.cdiv(C, NT)
    pid_m = pid // N_COL_GROUPS  # M-tile index (slow-varying)
    pid_g = pid % N_COL_GROUPS  # column-group index (fast-varying)
    c0 = pid_g * NT  # first block index in this column-group
    rm = pid_m * TILE_M + tl.arange(0, TILE_M)
    mask_m = rm < M
    col = c0 * GT + tl.arange(0, WIDTH)  # its WIDTH (= NT * GT) contiguous columns
    mask_col = col < C * GT
    x_slab = tl.load(
        x_ptr + rm[:, None] * stride_xm + col[None, :] * stride_xk,
        mask=mask_m[:, None] & mask_col[None, :],
        other=0.,
    )  # (TILE_M, NT * GT)
    rg = tl.arange(0, GT)
    if NT == 1:
        x_in = x_slab  # (TILE_M, GT); c0 < C always so its columns need no mask
        t = tl.load(t_ptr + c0 * stride_tc + rg[:, None] * stride_ti + rg[None, :] * stride_tj)  # (GT, GT)
    else:
        cc = c0 + tl.arange(0, NT)
        mask_cc = cc < C
        x_in = x_slab.reshape(TILE_M, NT, GT).permute(1, 0, 2)  # (NT, TILE_M, GT)
        t = tl.load(
            t_ptr + cc[:, None, None] * stride_tc + rg[None, :, None] * stride_ti + rg[None, None, :] * stride_tj,
            mask=mask_cc[:, None, None],
            other=0.,
        )  # (NT, GT, GT)
    if USE_DOT:
        y_in = tl.dot(x_in, t, out_dtype=tl.float32, input_precision='ieee')
    else:
        # y[.., m, j] = sum_i x[.., m, i] * t[.., i, j]  (rank-generic: NT==1 has no batch axis)
        y_in = (x_in.to(ACC).expand_dims(-1) * t.to(ACC).expand_dims(-3)).sum(axis=-2)
    if NT != 1:
        y_in = y_in.permute(1, 0, 2).reshape(TILE_M, NT * GT)  # (NT, TILE_M, GT) -> slab
    return y_in, rm, c0  # (TILE_M, NT * GT) slab + the tile's (rows, first block) coords


@triton.autotune(configs=TILE_CONFIGS, key=['C', 'GT', 'ISIZE', 'OSIZE'], prune_configs_by={'early_config_prune': prune_configs})
@triton.jit
def _kernel(
        x_ptr, t_ptr, y_ptr,
        M, C: tl.constexpr, GT: tl.constexpr,  # M is runtime: constexpr M -> a recompile per batch size
        stride_xm, stride_xk, stride_tc, stride_ti, stride_tj, stride_ym, stride_yk,
        ISIZE: tl.constexpr, OSIZE: tl.constexpr,  # OSIZE: autotune key only (out dtype); store casts
        TILE_M: tl.constexpr, WIDTH: tl.constexpr,
):
    NT: tl.constexpr = WIDTH // GT  # blocks per program (NT * GT = WIDTH cols)
    IS_FP64: tl.constexpr = ISIZE == 8  # fp64 is the only 8-byte input dtype
    ACC: tl.constexpr = tl.float64 if IS_FP64 else tl.float32
    USE_DOT: tl.constexpr = GT >= 16 and not IS_FP64  # tl.dot needs contraction dim K (= GT here) >= 16 --
    #   a Triton compiler check ("Input shapes should have M >= 1, N >= 1 and K >= 16"), not in the docs; M, N
    #   may be < 16. fp64 DOES compile (bit-exact) but has no tensor-core path on Ada (emulated -> no faster
    #   than the manual MAC, and hits the same smem ceiling), so GT<16 and fp64 both use the manual MAC below.
    slab, rm, c0 = load_transform(
        tl.program_id(0),
        x_ptr, t_ptr,
        M, C, GT,
        stride_xm, stride_xk,
        stride_tc, stride_ti, stride_tj,
        TILE_M, WIDTH,
        NT, USE_DOT, ACC,
    )
    mask_m = rm < M
    col = c0 * GT + tl.arange(0, WIDTH)  # the tile's WIDTH (= NT * GT) contiguous columns
    mask_col = col < C * GT
    tl.store(
        y_ptr + rm[:, None] * stride_ym + col[None, :] * stride_yk,
        slab,
        mask=mask_m[:, None] & mask_col[None, :],
    )  # auto-casts to the out dtype


def block_transform(
        x: torch.Tensor,
        transform: torch.Tensor,
        transposed: bool = False,
        out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    x:(..., K), transform: contiguous (C, GT, GT) -> (..., K). Per block: Y_c = X_c @ T_c^T.
      transposed=False (default): transform[c] = T_c; the kernel reads T_c^T via an index swap
          (no offline transpose / copy needed).
      transposed=True: transform[c] = T_c^T already (must be contiguous, e.g. `.transpose(-1, -2).contiguous()`).
    Pass a 2D (GT, GT) transform to apply the SAME matrix to every block (repeated/shared transform):
    the kernel just sees the flat buffer as (M * C, GT) via strides -- one block per kernel row, C=1
    (early_config_prune tiles that shape with WIDTH=GT). No reshape and no copies of the matrix; same
    result as a (C, GT, GT) tensor with the matrix repeated C times.
    GT is any power of two in [1, 128]; x and transform share dtype. out_dtype defaults to the
    input; pass fp32 with a 16-bit input to keep the full tensor-core fp32 accumulator.
    """
    K: int = x.size(-1)
    if transform.ndim == 2:  # repeated: kernel row = one block (C=1, row width GT)
        GT, GT2 = transform.shape
        C: int = 1
        assert K % GT == 0, 'last dim must be divisible by GT'
    else:  # per-block: kernel row = the whole K (= C * GT)
        C, GT, GT2 = transform.shape
        assert K == C * GT, 'transform must have K/GT blocks'
    assert GT == GT2 and 1 <= GT <= 128 and (GT & (GT - 1)) == 0, 'GT must be a power of two in [1, 128]'
    assert x.dtype == transform.dtype, 'x and transform must share dtype'
    out_dtype: torch.dtype = out_dtype or x.dtype
    assert out_dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64), \
        f'out_dtype must be a float type, got {out_dtype}'
    if x.dtype == torch.float64 and GT > 64:
        raise NotImplementedError(
            'fp64 transform is supported for GT <= 64; at GT=128 the manual fp64 working set '
            '(~2 MB/program) exceeds GPU resources. Use fp32 for GT=128.'
        )
    # transform is contiguous with inner strides (GT, 1), storing T (transposed=False) or T^T
    # (transposed=True). The kernel reads T^T, so swap (i, j) for the untransposed case.
    assert transform.is_contiguous(), 'transform must be a contiguous (C, GT, GT) or (GT, GT) tensor'
    sti, stj = (GT, 1) if transposed else (1, GT)
    xc: torch.Tensor = x.contiguous()
    row: int = C * GT  # the kernel's view is (numel / row, row): K per-block, GT repeated
    M: int = xc.numel() // row  # kernel rows: numel / K (per-block) or numel / GT (repeated)
    y: torch.Tensor = torch.empty_like(xc, dtype=out_dtype)  # the kernel uses flat strides; y keeps x's shape
    grid = lambda meta: (triton.cdiv(M, meta['TILE_M']) * triton.cdiv(C, meta['WIDTH'] // GT),)
    with device_guard(x.device):
        _kernel[grid](
            xc, transform, y,
            M=M, C=C, GT=GT,
            stride_xm=row, stride_xk=1,
            stride_tc=GT * GT, stride_ti=sti, stride_tj=stj,
            stride_ym=row, stride_yk=1,
            ISIZE=x.element_size(), OSIZE=y.element_size(),
        )
    return y


# =========================================================================== #
# Unit test + benchmark  (run from the source root:  CUDA_VISIBLE_DEVICES=<gpu> python -m triton_kernels.transform)
# =========================================================================== #


def _unit_test(
        device: torch.device = torch.device('cuda'),
) -> None:
    """
    Unit test: the block transform vs the PyTorch reference block_transform.apply_block_transform
    (Y_c = X_c @ T_c^T per block), computed in fp64 as the ground truth. Covers block size GT (every
    power of two in [1, 128]), input dtype (bf16 / fp16 / fp32 all GT; fp64 GT <= 64), independent output
    dtype, the transposed flag, and the repeated (shared 2D) transform.
    """

    from block_transform import apply_block_transform

    torch.manual_seed(seed=0)
    high_dtype: torch.dtype = torch.float64
    # dtype -> (rtol, atol), the SINGLE source every comparison below keys off (by output dtype) -- no loose
    # literals. The kernel rounds an fp32/fp64-accumulated result to the output dtype, so the error is
    # ~0.5 ulp(output) and ATOL is the meaningful bound; rtol is a non-binding safety net (it cannot be
    # tightened -- a block dot-product that cancels to ~0 carries a 0.5-ulp abs error == a huge REL error, so
    # the worst-case rel reaches ~1 for bf16, ~5 for fp32, no matter how exact the kernel is). atol is sized
    # to the highest-magnitude block in the sweep: bf16 = 0.5 ulp at |y|~11 (GT=2; measured 2.9e-2 over many
    # seeds vs 3e-2 -- so it also covers the GT=32 shapes' ~1.6e-2); fp32 = 0.5 ulp(fp32) (measured <=3.2e-6).
    dtypes: dict[torch.dtype, tuple[float, float]] = {
        torch.bfloat16: (3e-2, 3e-2),
        torch.float16: (5e-3, 5e-3),
        torch.float32: (1e-3, 1e-5),
        torch.float64: (1e-9, 1e-9),
    }
    GT: int = 32
    for M, K in (1000, 4096), (4096, 12288):  # main shapes (1000 exercises M-masking)
        x: torch.Tensor = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        transform: torch.Tensor = torch.randn(K // GT, GT, GT, dtype=torch.bfloat16, device=device) * GT ** -.5
        reference: torch.Tensor = apply_block_transform(x=x.to(dtype=high_dtype), transform=transform.to(dtype=high_dtype), high_dtype=high_dtype, round_dtype=high_dtype)
        assert block_transform(x=x, transform=transform).to(dtype=high_dtype).allclose(reference, *dtypes[x.dtype])
    x: torch.Tensor = torch.randn(3, 257, 2048, dtype=torch.bfloat16, device=device)  # ND input
    transform: torch.Tensor = torch.randn(2048 // GT, GT, GT, dtype=torch.bfloat16, device=device) * GT ** -.5
    y: torch.Tensor = block_transform(x=x, transform=transform)
    reference: torch.Tensor = apply_block_transform(x=x.to(dtype=high_dtype), transform=transform.to(dtype=high_dtype), high_dtype=high_dtype, round_dtype=high_dtype)
    assert y.shape == x.shape and y.to(dtype=high_dtype).allclose(reference, *dtypes[y.dtype])

    for dtype, (rtol, atol) in dtypes.items():  # block size GT x dtype; M=500 exercises masking
        for GT in 1, 2, 4, 8, 16, 32, 64, 128:
            if dtype is torch.float64 and GT > 32:
                continue  # documented limitation (fp64 manual working set exceeds sm_120's ~100KB smem at GT>=64; larger-smem arches reach GT=64 and only break at GT=128)
            x: torch.Tensor = torch.randn(500, GT * 32, dtype=dtype, device=device)
            transform: torch.Tensor = torch.randn((GT * 32) // GT, GT, GT, dtype=dtype, device=device) * GT ** -.5
            reference: torch.Tensor = apply_block_transform(x=x.to(dtype=high_dtype), transform=transform.to(dtype=high_dtype), high_dtype=high_dtype, round_dtype=high_dtype)
            assert block_transform(x=x, transform=transform).to(dtype=high_dtype).allclose(reference, rtol=rtol, atol=atol), (dtype, GT)

    for dtype in torch.bfloat16, torch.float16:  # 16-bit input -> fp32 output (full tensor-core accumulator)
        x: torch.Tensor = torch.randn(500, 4096, dtype=dtype, device=device)
        transform: torch.Tensor = torch.randn(4096 // 32, 32, 32, dtype=dtype, device=device) * 32 ** -.5
        y: torch.Tensor = block_transform(x=x, transform=transform, out_dtype=torch.float32)
        reference: torch.Tensor = apply_block_transform(x=x.to(dtype=high_dtype), transform=transform.to(dtype=high_dtype), high_dtype=high_dtype, round_dtype=high_dtype)
        assert y.dtype == torch.float32 and y.to(dtype=high_dtype).allclose(reference, *dtypes[y.dtype])

    x: torch.Tensor = torch.randn(500, 4096, dtype=torch.bfloat16, device=device)  # transposed flag: T_c == contiguous T_c^T
    transform: torch.Tensor = torch.randn(4096 // 32, 32, 32, dtype=torch.bfloat16, device=device) * 32 ** -.5
    y_untransposed: torch.Tensor = block_transform(x=x, transform=transform)
    y_transposed: torch.Tensor = block_transform(x=x, transform=transform.transpose(-1, -2).contiguous(), transposed=True)
    reference: torch.Tensor = apply_block_transform(x=x.to(dtype=high_dtype), transform=transform.to(dtype=high_dtype), high_dtype=high_dtype, round_dtype=high_dtype)
    assert y_untransposed.equal(y_transposed) and y_untransposed.to(dtype=high_dtype).allclose(reference, *dtypes[y_transposed.dtype])

    # repeated (shared) transform: a 2D (GT, GT) matrix applied to every block == the (C, GT, GT) path with
    # the matrix repeated C times (bit-identical on the tensor-core path GT >= 16, to an fp ulp otherwise)
    for dtype, (rtol, atol) in dtypes.items():
        for GT in [1, 8, 16] + ([] if dtype is torch.float64 else [128]):
            x: torch.Tensor = torch.randn(500, GT * 32, dtype=dtype, device=device)
            shared: torch.Tensor = torch.randn(GT, GT, dtype=dtype, device=device) * GT ** -.5
            per_block: torch.Tensor = shared.unsqueeze(0).repeat(32, 1, 1).contiguous()  # same matrix on all blocks
            y_2d, y_3d = block_transform(x=x, transform=shared), block_transform(x=x, transform=per_block)
            exact: bool = GT >= 16 and dtype is not torch.float64
            assert (y_2d.equal(y_3d) if exact else y_2d.to(dtype=high_dtype).allclose(y_3d.to(dtype=high_dtype), rtol=rtol, atol=atol))
            reference: torch.Tensor = apply_block_transform(x=x.to(dtype=high_dtype), transform=per_block.to(dtype=high_dtype), high_dtype=high_dtype, round_dtype=high_dtype)
            assert y_2d.to(dtype=high_dtype).allclose(reference, rtol=rtol, atol=atol), ('repeated', dtype, GT)

    print('Unit test passed.')


def _benchmark(
        device: torch.device = torch.device('cuda'),
) -> None:
    """
    Throughput (GB/s, read x + write y) of the block transform vs the PyTorch reference apply_block_transform.
    """

    from block_transform import apply_block_transform

    GT: int = 32
    print(f"{'K':>8} | {'triton':>11} | {'torch':>11}    (GB/s, read x + write y)")
    for K in (2 ** i for i in range(10, 16)):
        x: torch.Tensor = torch.randn(4096, K, dtype=torch.bfloat16, device=device)
        transform: torch.Tensor = torch.randn(K // GT, GT, GT, dtype=torch.bfloat16, device=device) * GT ** -.5
        gbps = lambda fn: 2 * x.numel() * x.element_size() * 1e-9 / (triton.testing.do_bench(fn) * 1e-3)  # read x + write y, GB/s
        print(f'{K:>8} | {gbps(lambda: block_transform(x=x, transform=transform)):>11.1f} | {gbps(lambda: apply_block_transform(x=x, transform=transform)):>11.1f}')


if __name__ == '__main__':
    _unit_test(device=torch.device('cuda'))
    _benchmark(device=torch.device('cuda'))
