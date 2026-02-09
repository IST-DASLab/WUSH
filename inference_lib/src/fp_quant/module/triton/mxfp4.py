from itertools import product

import torch
import triton
import triton.language as tl


def get_autotuning_config(configs, named_args, **kwargs):
    hadamard_dim = kwargs["hadamard_dim"]
    # Block size has to be chosen such, that BLOCK_SIZE // hadamard_dim is multiple of 16
    BLOCK_SIZES = [32 * 32, 64 * 32, 128 * 32, 256 * 32, 512 * 32]

    block_size_step = 16 * hadamard_dim

    for config in configs:
        num_warps = config.num_warps
        num_stages = config.num_stages
        for block_size in BLOCK_SIZES:
            if block_size // block_size_step > 0 and block_size % block_size_step == 0:
                yield triton.Config({"BLOCK_SIZE": block_size}, num_warps=num_warps, num_stages=num_stages) 


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages) 
        for num_stages, num_warps in product([1, 2], [1, 2, 4, 8])
    ],
    prune_configs_by={"early_config_prune": get_autotuning_config},
    key=[]
)
@triton.jit
def mxfp4_forward_kernel(
    x_ptr,
    hadamard_matrix_ptr,
    output_ptr,
    clip_mask_ptr,
    n_elements: tl.constexpr,
    hadamard_dim: tl.constexpr,
    group_size: tl.constexpr,
    gaussian_scale: tl.constexpr,
    quest: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WRITE_CLIP: tl.constexpr,
):
    offsets_hadamard = tl.arange(0, hadamard_dim * hadamard_dim)
    hadamard_matrix = tl.load(hadamard_matrix_ptr + offsets_hadamard).reshape(
        hadamard_dim, hadamard_dim
    )

    # load x
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x_flat = tl.load(x_ptr + offsets, mask=mask)

    # hadamard transform
    x = tl.reshape(x_flat, (BLOCK_SIZE // hadamard_dim, hadamard_dim))
    x_had = tl.dot(x, hadamard_matrix)

    # group
    x_had_grouped = tl.reshape(x_had, (BLOCK_SIZE // 32, 32))

    # scale
    if quest:
        mean_squared = (
            tl.sum(x_had_grouped * x_had_grouped, axis=-1, keep_dims=True) / 32
        )
        mean = tl.sum(x_had_grouped, axis=-1, keep_dims=True) / 32
        std = tl.sqrt(mean_squared - mean * mean)
        scales = gaussian_scale * std + 1e-8
        shared_exps = tl.exp2(tl.floor(tl.log2(scales)))
        x_had_scaled = x_had_grouped / shared_exps
    else:
        scales = tl.max(tl.abs(x_had_grouped), axis=-1, keep_dims=True)
        shared_exps = tl.exp2(tl.floor(tl.log2(scales)) - 2) / (3 / 4)
        x_had_scaled = x_had_grouped / shared_exps

    # quantize
    x_had_scaled_abs = tl.abs(x_had_scaled)
    x_had_scaled_sign = tl.where(
        x_had_scaled > 0,
        1,
        -1,
    )

    x_fp4 = (
        tl.where(
            x_had_scaled_abs > 5,
            6,
            tl.where(
                x_had_scaled_abs > 3.5,
                4,
                tl.where(
                    x_had_scaled_abs > 2.5,
                    3,
                    tl.where(
                        x_had_scaled_abs > 1.75,
                        2,
                        tl.where(
                            x_had_scaled_abs > 1.25,
                            1.5,
                            tl.where(
                                x_had_scaled_abs > 0.75,
                                1,
                                tl.where(
                                    x_had_scaled_abs > 0.25,
                                    0.5,
                                    0,
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
        * x_had_scaled_sign
    )
    if WRITE_CLIP:
        tl.store(
            clip_mask_ptr + offsets,
            tl.reshape(x_had_scaled_abs < 6, (BLOCK_SIZE,)),
            mask=mask,
        )

    # dequantize
    x_dequantized = x_fp4 * shared_exps

    # Reshape back to flat form for storage
    x_dequantized_flat = tl.reshape(x_dequantized, (BLOCK_SIZE,))

    # store
    tl.store(output_ptr + offsets, x_dequantized_flat, mask=mask)


@torch.compiler.disable()
def mxfp4_forward_kernel_wrapper(
    x,
    hadamard_matrix,
    return_clip_mask=False,
    gaussian_scale=3 / 4,
    quest=True,
):
    # Make sure inputs are contiguous
    x = x.contiguous()
    hadamard_matrix = hadamard_matrix.to(device=x.device).contiguous()

    # Create output tensor
    output = torch.empty_like(x)
    if return_clip_mask:
        clip_mask = torch.empty_like(x, dtype=torch.bool)
        clip_ptr, write_clip = clip_mask, True
    else:
        clip_mask = None
        clip_ptr, write_clip = x.new_empty(1), False

    # Get total number of elements and calculate grid for launching the kernel
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Launch optimized kernel
    with torch.device(x.device):
        mxfp4_forward_kernel[grid](
            x_ptr=x,
            hadamard_matrix_ptr=hadamard_matrix,
            output_ptr=output,
            clip_mask_ptr=clip_ptr,
            n_elements=n_elements,
            hadamard_dim=hadamard_matrix.shape[-1],
            group_size=32,
            gaussian_scale=gaussian_scale,
            quest=quest,
            WRITE_CLIP=write_clip,
        )

    return output, clip_mask
