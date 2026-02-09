import argparse
import copy
import functools
import itertools
import torch
from torch import nn
from transformers import PreTrainedModel
import scipy

try:
    import wandb
except ImportError:
    wandb = None

try:
    raise ImportError
    from fast_hadamard_transform import hadamard_transform
    get_normalized_hadamard_transform = lambda size, dtype=torch.float64, device=torch.device('cuda'): hadamard_transform(torch.eye(size, dtype=dtype, device=device), scale=size ** -.5)
except ImportError:
    get_normalized_hadamard_transform = lambda size, dtype=torch.float64, device=torch.device('cpu'): torch.as_tensor(scipy.linalg.hadamard(size), dtype=dtype, device=device) * size ** -.5

from .accumulate_hessian import accumulate_hessian
from ..utils.common_utils import clear_device_cache, to, maybe_first_element
from ..utils.model_utils import InputCollector, ForwardInterrupt
from ..utils.wush_utils import QuantLinear, BlockTransformQuantizer

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')


def rtn_e2m1(
        x: torch.Tensor,
        mode: str = 'even',
        return_packed: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    round to nearest fp4_e2m1
    x: (..., G)
    mode: str
    return_packed: bool
    returns: (..., G) fp, (..., G//2) or None
    """
    dtype, device = x.dtype, x.device
    grid: torch.Tensor = torch.as_tensor([-6., -4., -3., -2., -1.5, -1., -.5, -0., 0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=dtype, device=device)  # (16) fp
    grid_int: torch.Tensor = torch.as_tensor([-1, -2, -3, -4, -5, -6, -7, -8, 0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.uint8, device=device)  # (16) uint8
    inds: torch.Tensor = torch.bucketize(input=x, boundaries=grid, out_int32=False, right=False)  # (..., G) int64
    lo, hi = (inds - 1).clamp(min=0, max=15), inds.clamp(min=0, max=15)  # (..., G) int64, (..., G) int64
    g_lo, g_hi = grid[lo], grid[hi]  # (..., G) fp, (..., G) fp
    match mode:
        case 'even':
            pick_hi_eq: torch.Tensor = grid_int[hi] % 2 == 0  # (..., G) bool
        case 'zero':
            pick_hi_eq: torch.Tensor = grid_int[hi] >= 128  # (..., G) bool
        case 'down':
            pick_hi_eq: torch.Tensor = torch.zeros_like(x, dtype=torch.bool)  # (..., G) bool
        case 'up':
            pick_hi_eq: torch.Tensor = torch.ones_like(x, dtype=torch.bool)  # (..., G) bool
        case _:
            raise NotImplementedError
    pick_hi: torch.Tensor = (g_hi - x < x - g_lo) | (g_hi - x == x - g_lo) & pick_hi_eq  # (..., G) bool
    inds_picked: torch.Tensor = torch.where(pick_hi, hi, lo)  # (..., G) int64
    y: torch.Tensor = grid[inds_picked]  # (..., G) fp
    if not return_packed:
        return y, torch.empty((), dtype=torch.float4_e2m1fn_x2, device=device)  # (..., G) fp
    assert y.size(-1) % 2 == 0
    y_int: torch.Tensor = grid_int[inds_picked]  # (..., G) uint8
    y_int_packed: torch.Tensor = ((y_int[..., 1::2] & 0xF) << 4 | y_int[..., ::2] & 0xF).view(dtype=torch.float4_e2m1fn_x2)  # (..., G//2)
    return y, y_int_packed  # (... G//2)


def unpack_e2m1(
        x_packed: torch.Tensor,
        dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    unpack fp4_e2m1 from packed format
    x_packed: (..., G//2) float4_e2m1fn_x2
    dtype: torch.dtype
    returns: (..., G) fp
    """
    grid: torch.Tensor = torch.as_tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.,], dtype=dtype, device=x_packed.device)  # (16) fp
    x_int_packed: torch.Tensor = x_packed.view(dtype=torch.uint8)  # (..., G//2) uint8
    x_int: torch.Tensor = torch.stack([x_int_packed & 0xF, (x_int_packed >> 4) & 0xF], dim=-1).flatten(start_dim=-2)  # (..., G) unint8
    x: torch.Tensor = grid[x_int.to(dtype=torch.int64)]  # (..., G) fp
    return x  # (..., G) fp


def rtn_xxfp4(
        x: torch.Tensor,
        group_size: int = -1,
        scale_dtype: torch.dtype = torch.float64,
        scale_scale: float = 6.,
        global_scale: float = 1.,
        fp4_rounding_mode: str = 'even',
        existing_scale: torch.Tensor | None = None,
) -> dict:
    """
    round to nearest xxfp4 group format
    x: (..., C)
    group_size: int, G
    scale_dtype: torch.dtype
    global_scale: float
    scale_scale: float, clipping
    fp4_rounding_mode: str
    existing_scale: (..., C//G) pre-computed scale
    """
    dtype, high_dtype = x.dtype, torch.float64

    if group_size <= 0:
        group_size = x.size(-1)

    x_reshaped: torch.Tensor = x.to(dtype=high_dtype).unflatten(dim=-1, sizes=(-1, group_size))  # (..., C//G, G)

    if existing_scale is None:
        x_reshaped_abs_max: torch.Tensor = x_reshaped.abs().amax(dim=-1, keepdim=True)  # (..., C//G, 1)
        scale: torch.Tensor = x_reshaped_abs_max * global_scale / scale_scale  # (..., C//G, 1)
        scale_quantized: torch.Tensor = scale.to(dtype=scale_dtype)  # (..., C//G, 1)
    else:
        scale_quantized: torch.Tensor = existing_scale[..., None]  # (..., C//G, 1)

    scale_dequantized: torch.Tensor = scale_quantized.to(dtype=high_dtype)  # (..., C//G, 1)

    # x_reshaped_scaled: torch.Tensor = x_reshaped * scale_scale / x_reshaped_abs_max  # (..., C//G, G)
    x_reshaped_scaled: torch.Tensor = x_reshaped * global_scale / scale_dequantized  # (..., C//G, G)
    x_reshaped_scaled_dequantized, x_reshaped_scaled_quantized_packed = rtn_e2m1(x_reshaped_scaled, mode=fp4_rounding_mode, return_packed=True)  # (..., C//G, G), (..., C//G, G//2)
    x_reshaped_dequantized: torch.Tensor = x_reshaped_scaled_dequantized * scale_dequantized / global_scale  # (..., C//G, G)
    x_dequantized: torch.Tensor = x_reshaped_dequantized.flatten(start_dim=-2)  # (..., C)

    result: dict = {
        'fake_quant': x_dequantized.to(dtype=dtype),  # (..., C)
        'e2m1': x_reshaped_scaled_quantized_packed.flatten(start_dim=-2),  # (..., C//2)
        'scale_quant': scale_quantized[..., 0],  # (..., C//G)
        'global_scale': global_scale,
    }
    return result


def dequant_xxfp4(
        e2m1: torch.Tensor,
        scale_quant: torch.Tensor,
        global_scale: float,
        dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    dequantize xxfp4 group format
    e2m1: (..., C//2) float4_e2m1fn_x2
    scale_quant: (..., C//G)
    global_scale: float
    dtype: torch.dtype
    returns: (..., C)
    """
    high_dtype: torch.dtype = torch.float64
    scale_quantized: torch.Tensor = scale_quant[..., None]  # (..., C//G, 1)
    x_reshaped_scaled_quantized_packed: torch.Tensor = e2m1.unflatten(dim=-1, sizes=(scale_quantized.size(-2), -1))  # (..., C//G, G//2)
    x_reshaped_scaled_dequantized: torch.Tensor = unpack_e2m1(x_packed=x_reshaped_scaled_quantized_packed, dtype=high_dtype)  # (..., C//G, G)
    scale_dequantized: torch.Tensor = scale_quantized.to(dtype=high_dtype)  # (..., C//G, 1)
    x_reshaped_dequantized: torch.Tensor = x_reshaped_scaled_dequantized * scale_dequantized / global_scale  # (..., C//G, G)
    x_dequantized: torch.Tensor = x_reshaped_dequantized.flatten(start_dim=-2)  # (..., C)
    return x_dequantized.to(dtype=dtype)  # (..., C)


def gptq_babai_outer(
        parameter: torch.Tensor,
        hessian: torch.Tensor | None = None,
        inner_fn=lambda basis, y, **_: {'coefficient': torch.linalg.solve_triangular(basis, y, upper=True, left=True, unitriangular=False)},
        collate_fn=lambda *results, **_: torch.cat([r['coefficient'] for r in results], dim=-2).transpose(-2, -1),
        block_size: int = -1,
        force_rtn_outer: bool = False,
) -> dict | torch.Tensor:
    """
    GPTQ/LDLQ/Babai outer quantization
    parameter: (..., R, C)
    hessian: (..., C, C), None: RTN fallback
    inner_fn: function, inner quantization function
    collate_fn: function
    block_size: int, block size for processing channels, B
    force_rtn_outer: bool, RTN fallback
    returns: list[dict]
    """
    dtype, high_dtype = parameter.dtype, torch.float64
    device: torch.device = parameter.device

    n_vectors: int = parameter.size(-1)  # C
    if block_size <= 0:
        block_size = n_vectors

    x: torch.Tensor = parameter.transpose(-2, -1).to(dtype=high_dtype, device=device, copy=True, memory_format=torch.contiguous_format)  # (..., C, R), coefficient, updated in-place

    # decompose hessian matrix
    if hessian is not None:
        hessian_copy: torch.Tensor = hessian.to(dtype=high_dtype, device=device, copy=True, memory_format=torch.contiguous_format)  # (..., C, C)
    else:
        hessian_copy: torch.Tensor = torch.eye(n_vectors, dtype=high_dtype, device=device)  # (C, C)
    damp_ratio_cholesky: float = 1e-2
    diag_indices: torch.Tensor = torch.arange(n_vectors, dtype=torch.int64, device=device)  # (C)
    hessian_copy[..., diag_indices, diag_indices] += damp_ratio_cholesky * hessian_copy.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True)  # (..., C) <= (..., 1)
    max_try: int = 100
    basis: torch.Tensor = torch.empty_like(hessian_copy)  # (..., N=C, C)
    info: torch.Tensor = torch.empty(hessian_copy.shape[:-2], dtype=torch.int32, device=device)  # (...)
    while (max_try := max_try - 1) >= 0:
        torch.linalg.cholesky_ex(hessian_copy, upper=True, check_errors=False, out=(basis, info))  # basis: (..., N=C, C), upper triangular, column vectors, basis.t() @ basis = hessian_copy
        if not info.to(dtype=torch.bool).any():
            break
        hessian_copy[..., diag_indices, diag_indices] += damp_ratio_cholesky * hessian_copy.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True)  # (..., C) <= (..., 1)
    assert max_try >= 0, 'Hessian decomposition failed. Please try using more samples or increasing damp_ratio.'

    y: torch.Tensor = basis @ x  # (..., N=C, R), target (residual) column vectors, updated in-place

    # blockwise babai's nearest plane algorithm
    results: list[dict] = []
    for i1 in range((n_vectors - 1) // block_size * block_size, -1, -block_size):
        i2: int = i1 + block_size
        results.append(inner_fn(
            basis=basis[..., i1:i2, i1:i2],  # (..., N=B, B)
            y=y[..., i1:i2, :],  # (..., N=B, R)
        ))
        if not force_rtn_outer:
            x[..., i1:i2, :] = results[-1]['coefficient']  # (..., B, R)
        y[..., :i1, :] -= basis[..., :i1, i1:i2] @ x[..., i1:i2, :]  # (..., ?, R)
    results_collate = collate_fn(*results[::-1])
    # here we need to extract fake
    return results_collate


def block_transform_gptq_babai_inner(
        basis: torch.Tensor,
        y: torch.Tensor,
        quant_fn=lambda x, existing_scale=None, **_: {'fake_quant': x},
        basis_transform_fn=lambda basis, y, **_: torch.eye(basis.size(-1), dtype=basis.dtype, device=basis.device),
        force_rtn_inner: bool = False,
) -> dict:
    """
    inner function for gptq quantization with block transform
    y <=> basis @ x
      <=> basis @ transform.inv() @ (transform @ x)
      <=> basis @ transform_basis.t() @ x_fake_quant
      <=> basis @ x_coefficient
    basis: (..., N=B, B), upper triangular, column vectors
    y: (..., N=B, R), column vectors
    quant_fn: function to perform fake quantization
    basis_transform_fn: function to compute the transform for the basis
    force_rtn_inner: bool, RTN fallback
    returns: dict
    """
    # x: torch.Tensor = torch.linalg.solve_triangular(basis, y, upper=True, left=True, unitriangular=False)  # (..., B, R), basis @ x = y
    transform_basis: torch.Tensor = basis_transform_fn(basis=basis, y=y)  # (..., B, B)

    rotation, basis_new = torch.linalg.qr(basis @ transform_basis.transpose(-2, -1), mode='reduced')  # (..., N=B, N=B) orthogonal, (..., N=B, B) upper triangular column vectors
    y_new: torch.Tensor = rotation.transpose(-2, -1) @ y  # (..., N=B, R), updated in-place

    x_transformed: torch.Tensor = torch.linalg.solve_triangular(
        basis_new,  # (..., N=B, B), upper triangular, column vectors
        y_new,  # (..., N=B, R)
        upper=True,
        left=True,
        unitriangular=False,
    )  # (..., B, R), basis @ transform_basis.t() @ x_transformed = y <=> basis_new @ x_transformed = y_new

    result: dict = quant_fn(x_transformed.transpose(-2, -1), existing_scale=None)

    if force_rtn_inner:
        x_transformed: torch.Tensor = result['fake_quant'].transpose(-2, -1)  # (..., B, R)
    else:  # babai's nearest plane algorithm
        for i1 in range(basis_new.size(-1) - 1, -1, -1):
            i2: int = i1 + 1
            x_transformed[..., i1:i2, :] = y_new[..., i1:i2, :] / basis_new[..., i1:i2, i1:i2]
            result: dict = quant_fn(x_transformed.transpose(-2, -1), existing_scale=result['scale_quant'])
            x_transformed: torch.Tensor = result['fake_quant'].transpose(-2, -1)  # (..., B, R)
            y_new[..., :i1, :] -= basis_new[..., :i1, i1:i2] * x_transformed[..., i1:i2, :]

    x_coefficient: torch.Tensor = transform_basis.transpose(-2, -1) @ x_transformed  # (..., B, R)
    result['coefficient'] = x_coefficient  # (..., B, R)
    result['transform_basis'] = transform_basis  # (..., B, B)
    return result


def collate_xxfp4(
        *results,
        dtype: torch.dtype,
) -> dict:
    """
    collate packed xxfp4 results
    results: list[dict]
    dtype: torch.dtype
    returns: (..., R, C)
    """
    x: torch.Tensor = torch.cat([
        dequant_xxfp4(
            e2m1=result['e2m1'],
            scale_quant=result['scale_quant'],
            global_scale=result['global_scale'],
            dtype=dtype,
        ) for result in results], dim=-1)  # (..., R, C)
    quant_weight: torch.Tensor = torch.cat([result['e2m1'] for result in results], dim=-1)  # (..., C//2)
    scale_quant: torch.Tensor = torch.cat([result['scale_quant'] for result in results], dim=-1)  # (..., C//B)
    transform_basis: torch.Tensor = torch.stack([result['transform_basis'].to(dtype=dtype) for result in results], dim=-3)  # (..., C//B, B, B)
    collated_results: dict = {'fake_quant': x, 'transform_basis': transform_basis,'global_scale': results[0]['global_scale'], 'qweight': quant_weight.cpu(), 'scale_quant': scale_quant.cpu()}
    return collated_results


def get_transform(
        basis: torch.Tensor,
        y: torch.Tensor,
        transform_type: str = 'identity',
        transform_dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """

    basis: (..., B, B)
    y: (..., B, R)
    """
    dtype, high_dtype = basis.dtype, torch.float64
    device: torch.device = basis.device
    transform_block_size: int = basis.size(-1)

    match transform_type:
        case 'identity':
            transform: torch.Tensor = torch.eye(transform_block_size, dtype=high_dtype, device=device)  # (B, B)
        case 'random_rotation':
            transform: torch.Tensor = torch.linalg.qr(torch.randn(*basis.shape[:-2], transform_block_size, transform_block_size, dtype=dtype, device=device), mode='reduced').Q  # (..., B, B)
        case 'hadamard':
            transform: torch.Tensor = get_normalized_hadamard_transform(transform_block_size, dtype=high_dtype, device=device)  # (B, B)
        case 'wush' | 'wus':
            if transform_type == 'wus':
                hadamard: torch.Tensor = torch.eye(transform_block_size, dtype=high_dtype, device=device)  # (B, B)
            else:
                hadamard: torch.Tensor = get_normalized_hadamard_transform(transform_block_size, dtype=high_dtype, device=device)  # (B, B)
            v, s, uh = torch.linalg.svd(y, full_matrices=False)  # (..., B, B), (..., B), (..., B, R)
            s *= y.size(-1) ** -.5  # (..., B)
            damp_ratio_eigen: float = 1e-2
            s += s.mean(dim=-1, keepdim=True) * damp_ratio_eigen  # (..., B) <= (..., 1)
            transform: torch.Tensor = torch.linalg.solve_triangular(
                basis.transpose(-2, -1),  # (..., B, B)
                hadamard * s[..., None, :] ** .5 @ v.transpose(-2, -1),  # (..., B, B)
                upper=False,
                left=False,
                unitriangular=False,
            )  # (..., B, B), T_{hsvx}^{-\top}, transform = hadamard @ diag(s ** .5) @ v.t() @ basis.t().inv()
        case _:
            raise NotImplementedError

    transform: torch.Tensor = transform.to(dtype=transform_dtype).to(dtype=dtype)
    assert transform.isfinite().all()
    return transform


def block_transform(
        x: torch.Tensor,
        transform: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    compute x @ transform.t()
    x: (..., C)
    transform: (..., C//B, B, B)
    returns: (..., C)
    """
    dtype, high_dtype = x.dtype, torch.float64

    if transform is None:
        return x  # (..., C)

    x_transformed: torch.Tensor = (
        x.unflatten(dim=-1, sizes=(-1, 1, transform.size(-1))).to(dtype=high_dtype)  # (..., C//B, 1, B)
        @
        transform.transpose(-2, -1).to(dtype=high_dtype)  # (..., C//B, B, B)
    ).flatten(start_dim=-3).to(dtype=dtype)  # (..., C)
    return x_transformed  # (..., C)


@torch.no_grad()
def quantize_layers(
        *weights,
        hessian: torch.Tensor,
        quant_type: str,
        transform_class: str,
        force_rtn: bool = False,
) -> list[nn.Module]:
    """
    Quantize the weight matrix using GPTQ
    """
    weight: torch.Tensor = torch.cat(weights, dim=0)
    dtype, device = weight.dtype, weight.device

    match quant_type:
        case 'mxfp4':
            block_size: int = 32
            quant_fn = functools.partial(
                rtn_xxfp4,
                group_size=block_size,
                scale_dtype=torch.float8_e8m0fnu,
                scale_scale=4.,
                global_scale=3.,
                fp4_rounding_mode='even',
            )
        case 'nvfp4':
            block_size: int = 16
            quant_fn = functools.partial(
                rtn_xxfp4,
                group_size=block_size,
                scale_dtype=torch.float8_e4m3fn,
                scale_scale=6.,
                global_scale=10.,
                fp4_rounding_mode='even',
            )
        case _:
            raise NotImplementedError

    weight_quant_result: dict = gptq_babai_outer(
        parameter=weight,
        hessian=hessian,
        inner_fn=functools.partial(
            block_transform_gptq_babai_inner,
            quant_fn=quant_fn,
            basis_transform_fn=functools.partial(
                get_transform,
                transform_type=transform_class,
                transform_dtype=dtype,
            ),
            force_rtn_inner=force_rtn,  # use gptq
        ),
        collate_fn=functools.partial(collate_xxfp4, dtype=dtype),
        block_size=block_size,
        force_rtn_outer=force_rtn,  # use gptq
    )
    weight_transformed_dequantized: torch.Tensor = weight_quant_result['fake_quant']  # (R, C)
    transform_basis: torch.Tensor = weight_quant_result['transform_basis']  # (C//B, B, B)
    new_weights: list[torch.Tensor] = weight_transformed_dequantized.split([w.size(0) for w in weights], dim=0)
    quant_linear_layers: list[nn.Module] = [QuantLinear(
        weight=new_w,
        activation_quantizer=BlockTransformQuantizer(
            transform=transform_basis,
            quant_type=quant_type,
        ),
    ).to(device=device) for new_w in new_weights]
    new_quant_result: list[torch.Tensor] = weight_quant_result['qweight'].split([w.size(0) for w in weights], dim=0)
    new_scale_result: list[torch.Tensor] = weight_quant_result['scale_quant'].split([w.size(0) for w in weights], dim=0)
    quant_results=[{'transform_basis': transform_basis, 'qweight': new_quant_result[i], 'scale_quant': new_scale_result[i], 'global_scale': weight_quant_result['global_scale']} for i in range(len(weights))]  
    return quant_linear_layers, quant_results


class WUSH:
    def __init__(
        self,
        layer: nn.Module,
    ):
        assert isinstance(layer, nn.Linear)
        self.layer = layer
        self.hessian = None
        self.num_samples: int = 0

    # preparatory methods
    @torch.no_grad()
    def update(self, inp: torch.Tensor) -> None:
        """
        Update the estimate of Hessian matrix from a batch of data.

        Args:
            inp: batch of layer inputs
        """
        inp = inp.flatten(end_dim=-2)
        if self.hessian is None:
            self.hessian = torch.zeros(inp.size(-1), inp.size(-1), dtype=torch.float32, device=inp.device)
        batch_size: int = inp.size(0)
        self.hessian *= self.num_samples / (self.num_samples + batch_size)
        accumulate_hessian(self.hessian, inp.to(dtype=torch.float32) * (self.num_samples + batch_size) ** -.5)  # X^T X
        self.num_samples += batch_size


def wush_quantization_gptq(
    model: PreTrainedModel,
    calibration_data: list[torch.Tensor],
    args: argparse.Namespace, 
    device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    print("WUSH quantization started...")
    quantized_state_dict = {}
    non_quantized_state_dict = {}
    act_offload_device = "cpu" if args.cpu_offload_activations else device

    blocks = model.model.layers
    blocks[0] = InputCollector(blocks[0], cpu_offload=args.cpu_offload_activations)
    if args.cpu_offload_modules:
        model.get_input_embeddings().to(device)
        blocks[0] = blocks[0].to(device)

    for sample in calibration_data:
        try:
            with torch.no_grad():
                model(sample.to(device=device))
        except ForwardInterrupt:
            pass

    input_args = blocks[0].input_args
    input_kwargs = blocks[0].input_kwargs
    blocks[0] = blocks[0].module

    if args.cpu_offload_modules:
        model.get_input_embeddings().cpu()

    # Iterate over transformer blocks
    for block_idx, block in enumerate(blocks):
        print(f"Processing block {block_idx}...")
        if args.cpu_offload_modules:
            block.to(device)
        # 4. Create WUSH handles and hooks

        block_ref = copy.deepcopy(block)
        use_fused_layers: bool = False

        
        layer_sequences: list[list[str]] = [
                ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'],
                ['self_attn.o_proj'],
                ['mlp.gate_proj', 'mlp.up_proj'],
                ['mlp.down_proj'],
            ]  # for vllm (fused)

        if not use_fused_layers:
            layer_sequences: list[list[str]] = [[v] for v in itertools.chain.from_iterable(layer_sequences)]  # for non-vllm
            print("Layer sequences:", layer_sequences)
        
        wush_handles = {}
        hooks = {}
        for layer_name, *_ in layer_sequences:
            layer: nn.Module = block.get_submodule(layer_name)
            wush_handles[layer_name] = WUSH(layer)
            def update_handle_hook(name):
                def _hook(_, inp, __):
                    wush_handles[name].update(inp[0])
                return _hook
            hooks[layer_name] = layer.register_forward_hook(update_handle_hook(layer_name))

        # 5. Process calibration data
        output_ref = []
        for inp_args, inp_kwargs in zip(input_args, input_kwargs):
            with torch.no_grad():
                out_ref = block(*to(inp_args, device=device), **to(inp_kwargs, device=device))
            out_ref = maybe_first_element(out_ref).to(act_offload_device)
            output_ref.append(out_ref)
        # Remove hooks
        for hook in hooks.values():
            hook.remove()

        configs: list[tuple] = list(itertools.product(
            [('force_rtn', v) for v in [not args.gptq]],
            [('transform_class', v) for v in [args.transform_class]],
        ))
        print(configs)
        quant_layers: dict[tuple, dict[str, nn.Module]] = {config: {} for config in configs}
        real_quant_results: dict[tuple, dict[str, nn.Module]] = {config: {} for config in configs}

        for layer_sequence in layer_sequences:
            hessian = wush_handles[layer_sequence[0]].hessian
            weights = [block.get_submodule(layer_name).weight for layer_name in layer_sequence]
            for config in configs:
                qlinear_layers, quant_results = quantize_layers(*weights, hessian=hessian, quant_type=args.format + "4", **dict(config))
                quant_layers[config].update({layer_name: qlinear_layer for layer_name, qlinear_layer in zip(layer_sequence, qlinear_layers)})
                real_quant_results[config].update({layer_name: quant_result for layer_name, quant_result in zip(layer_sequence, quant_results)})


        l2_losses: dict[tuple, float] = {}
        for config in configs:
            for layer_name, qlinear_layer in quant_layers[config].items():
                block.set_submodule(layer_name, qlinear_layer)
            _losses = []
            for inp_args, inp_kwargs, out_ref in zip(input_args, input_kwargs, output_ref):
                with torch.no_grad():
                    out = block(*to(inp_args, device=device), **to(inp_kwargs, device=device))
                out = maybe_first_element(out)
                _losses.append(torch.linalg.vector_norm(out - out_ref.to(out.device), dtype=torch.float32).pow(2.) / out_ref.numel())
            l2_loss: float = torch.stack(_losses).mean().item()
            l2_losses[config] = l2_loss
            print('l2_loss', config, l2_loss)

        # find the min loss config
        best_config = min(l2_losses, key=l2_losses.get)
        best_quant_layers = quant_layers[best_config]
        best_real_quant_results = real_quant_results[best_config]
        print('best_config', best_config, 'l2_loss', l2_losses[best_config])

        for layer_name, qlinear_layer in best_quant_layers.items():
            block.set_submodule(layer_name, qlinear_layer)
            if args.export_quantized_model:
                    orig_dtype = model.config.torch_dtype if args.dtype == "auto" else args.dtype
                    weight_global_scale = act_global_scale = torch.tensor([3.0], dtype=torch.float32, device=device) if args.format == "mxfp" else torch.tensor([10.0], dtype=torch.float32, device=device)
                    transform_matrix = qlinear_layer.activation_quantizer.transform.to(dtype=orig_dtype).cpu()
                    transform_matrix = transform_matrix.view(-1, transform_matrix.shape[-1]).T.contiguous()
                    if args.export_quantized_model == "realquant":
                        quantized_state_dict[f"model.layers.{block_idx}.{layer_name}"] = {
                                "qweight": best_real_quant_results[layer_name]["qweight"].view(dtype=torch.uint8).cpu(),
                                "scales":best_real_quant_results[layer_name]["scale_quant"].view(torch.uint8).cpu(),
                                "forward_hadamard_matrix": transform_matrix,
                                "backward_hadamard_matrix": transform_matrix.clone(),
                                "weight_global_scale": weight_global_scale.clone(),
                                "act_global_scale": act_global_scale.clone()
                            }
                    else:
                        raise NotImplementedError("Only 'realquant' export is implemented.")
        # 8. Update activations
        for inp_args, inp_kwargs in zip(input_args, input_kwargs):
            with torch.no_grad():
                out = block(*to(inp_args, device=device), **to(inp_kwargs, device=device))
            out = maybe_first_element(out).to(act_offload_device)
            # change only first input argument
            if len(inp_args) > 0:
                inp_args[0].data = out
            elif "hidden_states" in inp_kwargs:
                inp_kwargs["hidden_states"] = out
            else:
                raise ValueError("Unsupported block input format.")
        
        if args.cpu_offload_modules:
            block.cpu()

        # 8. Clean-up
        del wush_handles
        del hooks
        clear_device_cache(garbage_collection=True)
        

    clear_device_cache(garbage_collection=True)
    print(model)
    return quantized_state_dict, non_quantized_state_dict
