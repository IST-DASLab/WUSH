import torch
import triton
from triton import language as tl

from triton_kernels.guard import device_guard


def _get_cuda_autotune_config() -> list[triton.Config]:
    return [
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 2}, num_warps=4, num_stages=3, num_ctas=1, maxnreg=None),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3, num_ctas=1, maxnreg=None),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4, num_ctas=1, maxnreg=None),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4, num_ctas=1, maxnreg=None),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4, num_ctas=1, maxnreg=None),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4, num_ctas=1, maxnreg=None),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4, num_ctas=1, maxnreg=None),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=2, num_stages=5, num_ctas=1, maxnreg=None),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=2, num_stages=5, num_ctas=1, maxnreg=None),
        # sm_100 (B200 / GB200): the 32-wide K tiles above under-feed the 5th-gen fp16 tensor cores; larger BLOCK_K (64/128) + a deeper pipeline saturate them.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=4, num_ctas=1, maxnreg=None),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3, num_ctas=1, maxnreg=None),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=4, num_ctas=1, maxnreg=None),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3, num_ctas=1, maxnreg=None),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=4, num_ctas=1, maxnreg=None),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3, num_ctas=1, maxnreg=None),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3, num_ctas=1, maxnreg=None),
    ]


@triton.autotune(
    configs=_get_cuda_autotune_config(),
    key=['size_hidden', 'size_batch', 'save_lower_only', 'compute_lower_only', 'size_meta_batch'],
    restore_value=['mat_hessian_ptr'],
)
@triton.jit
def accumulate_hessian_triton_kernel(
        mat_hessian_ptr,
        mat_input_ptr,
        size_hidden: int,
        size_batch: int,
        save_lower_only,
        compute_lower_only,
        size_meta_batch: int,  # autotune key
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
) -> None:
    a_ptr, b_ptr, c_ptr = mat_input_ptr, mat_input_ptr, mat_hessian_ptr
    M, N, K = size_hidden, size_hidden, size_batch
    stride_am, stride_ak = 1, size_hidden
    stride_bk, stride_bn = size_hidden, 1
    stride_cm, stride_cn = size_hidden, 1

    # Kernel for computing the matmul C = A x B. A has shape (M, K), B has shape (K, N) and C has shape (M, N)

    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    num_pid_mn = num_pid_m * num_pid_n
    meta_batch_id = pid // num_pid_mn
    # pid_within_batch = pid % num_pid_mn
    group_id = pid % num_pid_mn // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + pid % num_pid_in_group % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m

    is_upper = (pid_m + 1) * BLOCK_SIZE_M <= pid_n * BLOCK_SIZE_N
    if compute_lower_only and is_upper:
        return
    is_lower = pid_m * BLOCK_SIZE_M >= (pid_n + 1) * BLOCK_SIZE_N
    is_diag = not (is_lower or is_upper)

    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + meta_batch_id * (size_batch * size_hidden) \
             + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + meta_batch_id * (size_batch * size_hidden) \
             + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # TODO: (unknown reason) tl.load c here makes the kernel 2x slow
    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block of fp32 values for higher accuracy.
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.)
        # We accumulate along the K dimension.
        c = tl.dot(a, b, c)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + meta_batch_id * (size_hidden * size_hidden) \
             + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    c += tl.load(c_ptrs, mask=c_mask)
    tl.store(c_ptrs, c, mask=c_mask)

    if compute_lower_only and not (save_lower_only or is_diag):
        # Warning: BLOCK_SIZE_M:BLOCK_SIZE_N must be 1:n or n:1 for this kind of copying
        ct_ptrs = c_ptr + meta_batch_id * (size_hidden * size_hidden) \
                  + offs_cm[:, None] * stride_cn + offs_cn[None, :] * stride_cm
        tl.store(ct_ptrs, c, mask=c_mask)


def accumulate_hessian(
        mat_hessian: torch.Tensor,
        mat_input: torch.Tensor,
        save_lower_only: bool = False,
        compute_lower_only: bool = True,
        debug_mode: bool = False,
) -> torch.Tensor:
    """
    Accumulate the Hessian matrix (fp32) with the outer product of the input tensor (fp16 or bf16)
    mat_hessian: (size_meta_batch, size_hidden, size_hidden), fp32, the Hessian matrix to be accumulated, modified in-place and returned
    mat_input: (size_meta_batch, size_batch, size_hidden), fp16 or bf16, the input tensor
    save_lower_only: bool, whether to save the lower triangle only
    compute_lower_only: bool, whether to compute the lower triangle only (should be set to False only for debugging)
    debug_mode: bool, whether to use the baseline implementation without Triton
    """
    if debug_mode:
        return accumulate_hessian_baseline(mat_hessian, mat_input)

    assert compute_lower_only or not save_lower_only, 'compute_lower_only must be True when save_lower_only is True'
    assert mat_hessian.is_contiguous() and mat_input.is_contiguous()
    *meta_batch_dims, size_batch, size_hidden = mat_input.shape
    size_meta_batch: int = int(torch.as_tensor(meta_batch_dims).prod())
    grid = lambda meta: (
        size_meta_batch
        * triton.cdiv(size_hidden, meta['BLOCK_SIZE_M'])
        * triton.cdiv(size_hidden, meta['BLOCK_SIZE_N']),
    )
    # Instead of using a 2D grid, flatten the grid to 1D. This avoids the per-dimension limit.
    with device_guard(mat_input.device):
        accumulate_hessian_triton_kernel[grid](
            mat_hessian, mat_input,
            size_hidden, size_batch,
            save_lower_only,
            compute_lower_only,
            size_meta_batch,
        )
    return mat_hessian


def accumulate_hessian_baseline(mat_hessian: torch.Tensor, mat_input: torch.Tensor) -> torch.Tensor:
    mat_input = mat_input.view(-1, *mat_input.shape[-2:]).to(dtype=mat_hessian.dtype)
    return mat_hessian.view(-1, *mat_hessian.shape[-2:]).baddbmm_(
        mat_input.transpose(-2, -1), mat_input, beta=1, alpha=1,
    )  # mat_hessian += mat_input.t() @ mat_input


def _bad_baseline(mat_hessian: torch.Tensor, mat_input: torch.Tensor) -> torch.Tensor:
    mat_hessian += mat_input.transpose(-2, -1) @ mat_input
    return mat_hessian


def _unit_test(dtype: torch.dtype):
    torch.manual_seed(0)

    meta_batch_dims = (2,)
    size_batch, size_hidden = 16384, 4096

    mat_input = torch.randn(*meta_batch_dims, size_batch, size_hidden, device='cuda', dtype=dtype)

    torch_output = torch.randn(*meta_batch_dims, size_hidden, size_hidden, device='cuda', dtype=torch.float32)
    torch_output = torch_output + torch_output.transpose(-2, -1)
    bad_output = torch_output.clone()
    triton_output = torch_output.clone()
    triton_output_2 = torch_output.clone()
    triton_output_3 = torch_output.clone()

    accumulate_hessian_baseline(torch_output, mat_input)
    _bad_baseline(bad_output, mat_input)
    accumulate_hessian(triton_output, mat_input, save_lower_only=True, compute_lower_only=True)
    print(accumulate_hessian_triton_kernel.best_config)

    torch_output.tril_()
    bad_output.tril_()
    triton_output.tril_()

    accumulate_hessian(triton_output_2, mat_input, save_lower_only=False, compute_lower_only=True)
    assert (triton_output == triton_output_2.tril()).all() and (triton_output_2 == triton_output_2.transpose(-2, -1)).all()
    accumulate_hessian(triton_output_3, mat_input, save_lower_only=False, compute_lower_only=False)
    assert (triton_output == triton_output_3.tril()).all() and (triton_output_3 == triton_output_3.transpose(-2, -1)).all()

    # the triton kernel accumulates in fp32, so it must land CLOSER to the fp32 reference than the naive 16-bit @
    # (_bad_baseline computes the gram in the input's 16-bit dtype). Typical margin: ~5x (fp16), ~47x (bf16).
    err_bad = (bad_output - torch_output).abs().mean().item()
    err_triton = (triton_output - torch_output).abs().mean().item()
    print(f'mean|err| vs fp32 reference -- 16-bit baseline {err_bad:.3e} | triton {err_triton:.3e}')
    assert err_triton < err_bad, f'triton kernel ({err_triton:.3e}) not more precise than the 16-bit baseline ({err_bad:.3e})'

    print('Unit test passed.')


def _benchmark(dtype: torch.dtype):
    from matplotlib import pyplot as plt

    configs = [triton.testing.Benchmark(
        x_names=['N', 'K'],  # Argument names to use as an x-axis for the plot
        x_vals=[2 ** i for i in range(8, 15)],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        line_vals=['pytorch', 'bad', 'triton', 'triton_l'],  # Label name for the lines
        line_names=['PyTorch (FP32)', 'PyTorch (FP16)', 'Triton', 'Triton (Lower)'],  # Line styles
        plot_name='matmul-performance',  # Name for the plot, used also as a file name for saving the plot.
        args={},
        xlabel='N',
        ylabel='TFLOPS',
        x_log=True,
        y_log=True,
        styles=[('#1f77b4', '-'), ('#ff7f0e', '-'), ('#2ca02c', '-'), ('#2ca02c', '--')],
    )]
    # ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    @triton.testing.perf_report(configs)
    def __benchmark(N, K, provider):
        meta_batch_dims = ()
        a = torch.randn(*meta_batch_dims, K, N, device='cuda', dtype=dtype)
        c = torch.randn(*meta_batch_dims, N, N, device='cuda', dtype=torch.float32)
        quantiles = [.5, .2, .8]
        match provider:
            case 'pytorch':
                ms, min_ms, max_ms = triton.testing.do_bench(
                    lambda: accumulate_hessian_baseline(c, a),
                    quantiles=quantiles,
                )
            case 'bad':
                ms, min_ms, max_ms = triton.testing.do_bench(lambda: _bad_baseline(c, a), quantiles=quantiles)
            case 'triton':
                ms, min_ms, max_ms = triton.testing.do_bench(
                    lambda: accumulate_hessian(c, a, save_lower_only=False, compute_lower_only=False),
                    quantiles=quantiles,
                )
            case 'triton_l':
                ms, min_ms, max_ms = triton.testing.do_bench(
                    lambda: accumulate_hessian(c, a, save_lower_only=False, compute_lower_only=True),
                    quantiles=quantiles,
                )
            case _:
                raise NotImplementedError
        perf = lambda ms: 2. * N * N * K * torch.as_tensor(meta_batch_dims).prod().item() * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)

    result_dfs = __benchmark.run(show_plots=False, print_data=True, return_df=True)
    plt.grid()
    plt.show()
    return result_dfs


if __name__ == '__main__':
    _unit_test(dtype=torch.float16)
    _unit_test(dtype=torch.bfloat16)
    _benchmark(dtype=torch.float16)
    _benchmark(dtype=torch.bfloat16)
