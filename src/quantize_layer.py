import functools

import torch

from block_transform import apply_block_transform, apply_block_transform_gram, get_normalized_hadamard_transform, get_random_orthogonal_transform, get_wush_transform
from compute_gram import dampen_gram, get_diag_block, get_gram, invert_gram
from quantize_fp4 import dequant_xxfp4, rtn_xxfp4


def rtn_inner(
        *,
        x: torch.Tensor,  # (..., d_out, d)
        quant_fn=lambda x, **_: {'fake_quant': x, 'scale_quant': None},
        **_,
) -> dict:
    result: dict = quant_fn(x, existing_scale=None)
    result['coefficient'] = result['fake_quant']  # (..., d_out, d)
    return result


def gptq_inner(
        *,
        x: torch.Tensor,  # (..., d_out, d)
        hessian_sub_space: torch.Tensor,  # (..., d, d)
        quant_fn=lambda x, **_: {'fake_quant': x, 'scale_quant': None},
        force_rtn_inner: bool = False,
        **_,
) -> dict:
    d: int = x.size(-1)
    l: torch.Tensor = torch.linalg.cholesky_ex(
        invert_gram(gram=hessian_sub_space, inplace=False),  # (..., d, d)
        upper=False,
        check_errors=True,
    ).L  # (..., d, d), l @ l.t() = hessian.inverse()
    l /= l.diagonal(offset=0, dim1=-2, dim2=-1)[..., None, :].clone()  # (..., d, d)
    result: dict = quant_fn(x, existing_scale=None)  # group scales are computed once from the initial block and frozen during error propagation (standard GPTQ group behavior)
    x = x.clone()  # (..., d_out, d)
    for i1 in range(d):
        i2: int = i1 + 1
        result: dict = quant_fn(x, existing_scale=result['scale_quant'])
        error: torch.Tensor = (x if force_rtn_inner else result['fake_quant'])[..., :, i1:i2] - x[..., :, i1:i2]  # (..., d_out, 1)
        x[..., :, i1:] += error * l[..., i1:, i1:i2].transpose(-2, -1)  # (..., d_out, d - i1)
    result['coefficient'] = result['fake_quant']  # (..., d_out, d)
    return result


def babai_inner(
        *,
        x: torch.Tensor,  # (..., d_out, d)
        hessian_sub_space: torch.Tensor,  # (..., d, d)
        quant_fn=lambda x, **_: {'fake_quant': x, 'scale_quant': None},
        force_rtn_inner: bool = False,
        **_,
) -> dict:
    d: int = x.size(-1)
    l: torch.Tensor = torch.linalg.cholesky_ex(
        hessian_sub_space,  # (..., d, d)
        upper=False,
        check_errors=True,
    ).L  # (..., d, d), l @ l.t() = hessian
    result: dict = quant_fn(x, existing_scale=None)  # group scales are computed once from the initial block and frozen during error propagation
    y: torch.Tensor = x @ l  # (..., d_out, d)
    x = torch.full_like(x, torch.nan)  # (..., d_out, d)
    for i1 in range(d - 1, -1, -1):
        i2: int = i1 + 1
        x[..., :, i1:i2] = y[..., :, i1:i2] / l[..., i1:i2, i1:i2]  # (..., d_out, 1)
        result: dict = quant_fn(x, existing_scale=result['scale_quant'])
        y[..., :, :i1] -=(x if force_rtn_inner else result['fake_quant'])[..., :, i1:i2] * l[..., i1:i2, :i1]  # (..., d_out, i1)
    result['coefficient'] = result['fake_quant']  # (..., d_out, d)
    return result


def transform_inner(
        *,
        x: torch.Tensor,  # (..., d_out, d)
        hessian_sub_space: torch.Tensor,  # (..., d, d)
        hessian_full_space: torch.Tensor,  # (..., d, d)
        inner_fn=lambda x, **_: {'coefficient': x},
        get_transform_fn=lambda gram_activation, **_: torch.eye(gram_activation.size(-1), dtype=gram_activation.dtype, device=gram_activation.device).expand(*gram_activation.shape[:-2], -1, -1).contiguous(),
        use_full_space_hessian: bool = True,
        dampen_ratio: float = 0.,
        **_,
) -> dict:
    d_out: int = x.size(-2)
    # both grams are computed eagerly to keep the get_transform_fn interface uniform, although only the WUSH transform consumes their values;
    # full_space: diagonal block of the layer Hessian (weight-activation objective); sub_space: GPTQ effective block Hessian (weight-only objective, already dampened by the outer function)
    transform: torch.Tensor = get_transform_fn(
        gram_weight=dampen_gram(gram=get_gram(x, dtype=None) / d_out, ratio=dampen_ratio, inplace=True),  # (..., d, d)
        gram_activation=dampen_gram(gram=hessian_full_space, ratio=dampen_ratio, inplace=False) if use_full_space_hessian else hessian_sub_space,  # (..., d, d)
    )[..., None, :, :]  # (..., 1, d, d)
    x_transformed: torch.Tensor = apply_block_transform(
        x=x,  # (..., d_out, d)
        transform=transform,  # (..., 1, d, d)
        is_inverse_transpose=True,
        high_dtype=None,
        round_dtype=None,
    )  # (..., d_out, d)
    hessian_sub_space_transformed: torch.Tensor = apply_block_transform_gram(
        gram=hessian_sub_space,  # (..., d, d)
        transform=transform,  # (..., 1, d, d)
        is_inverse_transpose=False,
    )  # (..., d, d)
    hessian_full_space_transformed: torch.Tensor = apply_block_transform_gram(
        gram=hessian_full_space,  # (..., d, d)
        transform=transform,  # (..., 1, d, d)
        is_inverse_transpose=False,
    )  # (..., d, d)
    result: dict = inner_fn(
        x=x_transformed,  # (..., d_out, d)
        hessian_sub_space=hessian_sub_space_transformed,  # (..., d, d)
        hessian_full_space=hessian_full_space_transformed,  # (..., d, d)
    )
    x_coefficient: torch.Tensor = apply_block_transform(
        x=result['coefficient'],  # (..., d_out, d)
        transform=transform.transpose(-2, -1),  # (..., 1, d, d)
        is_inverse_transpose=False,
        high_dtype=None,
        round_dtype=None,
    )  # (..., d_out, d)
    result['coefficient'], result['transform'] = x_coefficient, transform  # (..., d_out, d), (..., 1, d, d)
    return result


def get_transform(
        *,
        gram_weight: torch.Tensor,  # (..., d, d)
        gram_activation: torch.Tensor,  # (..., d, d)
        transform_type: str = 'I',
        round_dtype: torch.dtype | None = None,
        **_,
) -> torch.Tensor:
    dtype, device = gram_activation.dtype, gram_activation.device  # gram_activation also sets the dtype/device/batch_dims for the gram-free transform types
    round_dtype = dtype if round_dtype is None else round_dtype
    *batch_dims, _, d = gram_activation.shape
    match transform_type.upper():
        case 'I':
            transform: torch.Tensor = torch.eye(d, dtype=dtype, device=device).expand(*batch_dims, -1, -1).contiguous()  # (..., d, d)
        case 'R':
            transform: torch.Tensor = get_random_orthogonal_transform(
                *batch_dims,
                size=d,
                dtype=dtype,
                device=device,
                enforce_rotation=False,
                high_dtype=None,
                round_dtype=round_dtype,
            )  # (..., d, d)
        case 'H':
            transform: torch.Tensor = get_normalized_hadamard_transform(
                d,
                dtype=dtype,
                device=device,
            ).to(dtype=round_dtype).to(dtype=dtype).expand(*batch_dims, -1, -1).contiguous()  # (..., d, d)
        case 'WUSH':
            transform: torch.Tensor = get_wush_transform(
                gram_weight=gram_weight,  # (..., d, d)
                gram_activation=gram_activation,  # (..., d, d)
                preserve_norm='balanced',
                use_hadamard=True,
                high_dtype=None,
                round_dtype=round_dtype,
            )
        case _:
            raise NotImplementedError
    return transform  # (..., d, d)


def rtn_outer(
        *,
        x: torch.Tensor,  # (..., d_out, d_in)
        hessian: torch.Tensor,  # (..., d_in, d_in)
        block_size: int,
        inner_fn=lambda x, **_: {'coefficient': x},
        collate_fn=lambda *results, **_: {'coefficient': torch.cat([r['coefficient'] for r in results], dim=-1)},
        dampen_ratio: float = 0.,
        **_,
) -> dict:
    d_in: int = x.size(-1)
    hessian_dampened: torch.Tensor = dampen_gram(gram=hessian, ratio=dampen_ratio, inplace=False)  # (..., d_in, d_in)
    results: list[dict] = []
    for i1 in range(0, d_in, block_size):
        i2: int = i1 + block_size
        x_block: torch.Tensor = x[..., :, i1:i2]  # (..., d_out, d)
        result: dict = inner_fn(
            x=x_block,  # (..., d_out, d)
            hessian_sub_space=hessian_dampened[..., i1:i2, i1:i2],  # (..., d, d)
            hessian_full_space=hessian[..., i1:i2, i1:i2],  # (..., d, d)
        )
        results.append(result)
    return collate_fn(*results)


def rtn_outer_parallel(
        *,
        x: torch.Tensor,  # (..., d_out, d_in)
        hessian: torch.Tensor,  # (..., d_in, d_in)
        block_size: int,
        inner_fn=lambda x, **_: {'coefficient': x},
        collate_fn=lambda result, **_: {'coefficient': result['coefficient'].transpose(-3, -2).flatten(start_dim=-2, end_dim=-1)},
        dampen_ratio: float = 0.,
        **_,
) -> dict:
    hessian_dampened: torch.Tensor = dampen_gram(gram=hessian, ratio=dampen_ratio, inplace=False)  # (..., d_in, d_in)
    x_blocks: torch.Tensor = x.unflatten(dim=-1, sizes=(-1, block_size)).transpose(-3, -2)  # (..., d_in // d, d_out, d)
    result: dict = inner_fn(
        x=x_blocks,  # (..., d_out, d)
        hessian_sub_space=get_diag_block(matrix=hessian_dampened, size=block_size),  # (..., d_in // d, d, d)
        hessian_full_space=get_diag_block(matrix=hessian, size=block_size),  # (..., d_in // d, d, d)
    )
    return collate_fn(result)


def gptq_outer(
        *,
        x: torch.Tensor,  # (..., d_out, d_in)
        hessian: torch.Tensor,  # (..., d_in, d_in)
        block_size: int,
        inner_fn=lambda x, **_: {'coefficient': x},
        collate_fn=lambda *results, **_: {'coefficient': torch.cat([r['coefficient'] for r in results], dim=-1)},
        dampen_ratio: float = 0.,
        force_rtn_outer: bool = False,
        **_,
) -> dict:
    d_in: int = x.size(-1)
    hessian_dampened: torch.Tensor = dampen_gram(gram=hessian, ratio=dampen_ratio, inplace=False)  # (..., d_in, d_in)
    l: torch.Tensor = torch.linalg.cholesky_ex(
        invert_gram(gram=hessian_dampened, inplace=False),  # (..., d_in, d_in)
        upper=False,
        check_errors=True,
    ).L  # (..., d_in, d_in), l @ l.t() = hessian_dampened.inverse()
    results: list[dict] = []
    x = x.clone()  # (..., d_out, d_in)
    for i1 in range(0, d_in, block_size):
        i2: int = i1 + block_size
        x_block: torch.Tensor = x[..., :, i1:i2]  # (..., d_out, d)
        result: dict = inner_fn(
            x=x_block,  # (..., d_out, d)
            hessian_sub_space=hessian_dampened[..., i1:i2, i1:i2] if force_rtn_outer else l[..., i1:i2, i1:i2].cholesky_inverse(upper=False),  # (..., d, d), (l_ii @ l_ii.t()).inverse() = effective blockwise Hessian after eliminating the preceding blocks
            hessian_full_space=hessian[..., i1:i2, i1:i2],  # (..., d, d)
        )
        error: torch.Tensor = (x_block if force_rtn_outer else result['coefficient']) - x_block  # (..., d_out, d)
        ll: torch.Tensor = torch.linalg.solve_triangular(
            l[..., i1:i2, i1:i2],  # (..., d, d)
            l[..., i1:, i1:i2],  # (..., d_in - i1, d)
            upper=False,
            left=False,
            unitriangular=False,
        )  # (..., d_in - i1, d)
        x[..., :, i1:] += error @ ll.transpose(-2, -1)  # (..., d_out, d_in - i1)
        results.append(result)
    return collate_fn(*results)


def babai_outer(
        *,
        x: torch.Tensor,  # (..., d_out, d_in)
        hessian: torch.Tensor,  # (..., d_in, d_in)
        block_size: int,
        inner_fn=lambda x, **_: {'coefficient': x},
        collate_fn=lambda *results, **_: {'coefficient': torch.cat([r['coefficient'] for r in results], dim=-1)},
        dampen_ratio: float = 0.,
        force_rtn_outer: bool = False,
        **_,
) -> dict:
    d_in: int = x.size(-1)
    hessian_dampened: torch.Tensor = dampen_gram(gram=hessian, ratio=dampen_ratio, inplace=False)  # (..., d_in, d_in)
    l: torch.Tensor = torch.linalg.cholesky_ex(
        hessian_dampened,  # (..., d_in, d_in)
        upper=False,
        check_errors=True,
    ).L  # (..., d_in, d_in), l @ l.t() = hessian_dampened
    results: list[dict] = []
    y: torch.Tensor = x @ l  # (..., d_out, d_in)
    for i1 in range((d_in - 1) // block_size * block_size, -1, -block_size):
        i2: int = i1 + block_size
        x_block: torch.Tensor = torch.linalg.solve_triangular(
            l[..., i1:i2, i1:i2],  # (..., d, d)
            y[..., :, i1:i2],  # (..., d_out, d)
            upper=False,
            left=False,
            unitriangular=False,
        )  # (..., d_out, d)
        result: dict = inner_fn(
            x=x_block,  # (..., d_out, d)
            hessian_sub_space=hessian_dampened[..., i1:i2, i1:i2] if force_rtn_outer else get_gram(l[..., i1:i2, i1:i2].transpose(-2, -1), dtype=None),  # (..., d, d), l_ii @ l_ii.t() = effective blockwise Hessian
            hessian_full_space=hessian[..., i1:i2, i1:i2],  # (..., d, d)
        )
        y[..., :, :i1] -= (x_block if force_rtn_outer else result['coefficient']) @ l[..., i1:i2, :i1]  # (..., d_out, i1)
        results.append(result)
    return collate_fn(*results[::-1])


def collate_xxfp4(
        *results,  # *list[dict], packed xxfp4 results
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
) -> dict:
    e2m1: torch.Tensor = torch.cat([result['e2m1'] for result in results], dim=-1)  # (..., d_out, d_in // 2)
    scale_quant: torch.Tensor = torch.cat([result['scale_quant'] for result in results], dim=-1)  # (..., d_out, d_in // d)
    global_scale: float = results[0]['global_scale']
    fake_quant: torch.Tensor = torch.cat([result['fake_quant'] for result in results], dim=-1)  # (..., d_out, d_in)
    transform: torch.Tensor = torch.cat([result['transform'] for result in results], dim=-3)  # (..., d_in // d, d, d)
    coefficient: torch.Tensor = torch.cat([result['coefficient'] for result in results], dim=-1)  # (..., d_out, d_in)
    collated_results: dict = {
        'transform': transform.to(dtype=dtype, device=device),  # (..., d_in // d, d, d)
        'e2m1': e2m1.to(device=device),  # (..., d_out, d_in // 2)
        'scale_quant': scale_quant.to(device=device),  # (..., d_out, d_in // d)
        'global_scale': global_scale,
        'fake_quant': fake_quant.to(dtype=dtype, device=device),  # (..., d_out, d_in)
        'coefficient': coefficient.to(dtype=dtype, device=device),  # (..., d_out, d_in)
    }
    return collated_results


def collate_parallel_xxfp4(
        result: dict,  # dict, packed xxfp4 results
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
) -> dict:
    e2m1: torch.Tensor = result['e2m1'].transpose(-3, -2).view(dtype=torch.uint8).flatten(start_dim=-2, end_dim=-1).view(dtype=torch.float4_e2m1fn_x2)  # (..., d_out, d_in // 2), round-trip via uint8 because copy ops are not implemented for fp4
    scale_quant: torch.Tensor = result['scale_quant'].transpose(-3, -2).flatten(start_dim=-2, end_dim=-1)  # (..., d_out, d_in // d)
    global_scale: float = result['global_scale']
    fake_quant: torch.Tensor = result['fake_quant'].transpose(-3, -2).flatten(start_dim=-2, end_dim=-1)  # (..., d_out, d_in)
    transform: torch.Tensor = result['transform'].squeeze(dim=-3)  # (..., d_in // d, d, d)
    coefficient: torch.Tensor = result['coefficient'].transpose(-3, -2).flatten(start_dim=-2, end_dim=-1)  # (..., d_out, d_in)
    collated_results: dict = {
        'transform': transform.to(dtype=dtype, device=device),  # (..., d_in // d, d, d)
        'e2m1': e2m1.to(device=device),  # (..., d_out, d_in // 2)
        'scale_quant': scale_quant.to(device=device),  # (..., d_out, d_in // d)
        'global_scale': global_scale,
        'fake_quant': fake_quant.to(dtype=dtype, device=device),  # (..., d_out, d_in)
        'coefficient': coefficient.to(dtype=dtype, device=device),  # (..., d_out, d_in)
    }
    return collated_results


def quantize_layer(
        x: torch.Tensor,
        hessian: torch.Tensor,
        quantize_format_name: str,
        method_name: str = 'GPTQ',
        transform_type: str = 'WUSH',
        quantize_side_name: str = 'WA',
        dampen_ratio: float = 1e-2,
        round_dtype: torch.dtype | None = None,
):
    dtype, high_dtype = x.dtype, hessian.dtype  # the hessian dtype sets the compute precision (pass fp32/fp64)
    round_dtype = dtype if round_dtype is None else round_dtype

    match quantize_format_name.upper():
        case 'MXFP4':
            block_size: int = 32
            scale_scale: float = 4.
            global_scale: float = 1. / 3.
            scale_dtype: torch.dtype = torch.float8_e8m0fnu
        case 'NVFP4':
            block_size: int = 16
            scale_scale: float = 6.
            # global_scale: float = (x.abs().amax().item() + torch.finfo(dtype).eps) / (torch.finfo(torch.float8_e4m3fn).max * 6.)  # NVFP4 per-tensor global scale: absmax/(448*6) so the largest group AbsMax maps to the e4m3 maximum; the e4m3 clamp in rtn_xxfp4 absorbs any AbsMax growth caused by the transforms or GPTQ updates
            global_scale: float = .1
            scale_dtype: torch.dtype = torch.float8_e4m3fn
        case _:
            raise NotImplementedError
    quant_fn = functools.partial(
        rtn_xxfp4,
        group_size=block_size,
        scale_dtype=scale_dtype,
        scale_scale=scale_scale,
        global_scale=global_scale,
        fp4_rounding_mode='even',
        high_dtype=high_dtype,
        round_dtype=round_dtype,
    )

    match method_name.upper():
        case 'RTN':
            collate_fn, outer_fn, inner_fn = collate_parallel_xxfp4, rtn_outer_parallel, rtn_inner
        case 'GPTQ':
            collate_fn, outer_fn, inner_fn = collate_xxfp4, gptq_outer, gptq_inner
        case 'BABAI':
            collate_fn, outer_fn, inner_fn = collate_xxfp4, babai_outer, babai_inner
        case _:
            raise NotImplementedError

    match quantize_side_name.upper():
        case 'W':
            use_full_space_hessian: bool = False
        case 'WA':
            use_full_space_hessian: bool = True
        case _:
            raise NotImplementedError

    result: dict = outer_fn(
        x=x.to(dtype=high_dtype),  # (..., d_out, d_in)
        hessian=hessian,  # (..., d_in, d_in)
        block_size=block_size,
        inner_fn=functools.partial(
            transform_inner,
            inner_fn=functools.partial(
                inner_fn,
                quant_fn=quant_fn,
            ),
            get_transform_fn=functools.partial(
                get_transform,
                transform_type=transform_type,
                round_dtype=round_dtype,
            ),
            use_full_space_hessian=use_full_space_hessian,
            dampen_ratio=dampen_ratio,
        ),
        collate_fn=functools.partial(
            collate_fn,
            dtype=dtype,
            device=None,
        ),
        dampen_ratio=dampen_ratio,
    )
    result['scale_scale'] = scale_scale  # group-scale clip divisor (4 mxfp4 / 6 nvfp4); needed to fake-quant activations the same way
    return result


def _unit_test(
        device: torch.device = torch.device('cuda'),
) -> None:
    torch.manual_seed(seed=0)
    dtype, high_dtype = torch.float32, torch.float64

    def is_power_of_two(d: int) -> bool:
        return d > 0 and (d & (d - 1)) == 0

    batch_dims: tuple[int, ...] = 3, 5
    d: int = 32
    d_batch, d_in, d_out = 59, d * 7, 67
    assert is_power_of_two(d) and d_in % d == 0 and d <= min(d_batch, d_out)

    dampen_ratio: float = 1e-2

    def get_random_data(*batch_dims, m: int, n: int) -> torch.Tensor:
        k: int = min(m, n)
        return torch.einsum(
            '...ik,...k,...jk->...ij',
            get_random_orthogonal_transform(*batch_dims, size=m, dtype=high_dtype, device=device, enforce_rotation=False, high_dtype=None, round_dtype=None)[..., :k],  # (..., m, k)
            torch.randn(*batch_dims, k, dtype=high_dtype, device=device).exp(),  # (..., k)
            get_random_orthogonal_transform(*batch_dims, size=n, dtype=high_dtype, device=device, enforce_rotation=False, high_dtype=None, round_dtype=None)[..., :k],  # (..., n, k)
        ).to(dtype=dtype).to(dtype=high_dtype)

    weight: torch.Tensor = get_random_data(*batch_dims, m=d_out, n=d_in)  # (..., d_out, d_in)
    activation: torch.Tensor = get_random_data(*batch_dims, m=d_batch, n=d_in)  # (..., d_batch, d_in)

    hessian: torch.Tensor = get_gram(activation, dtype=None) / d_batch  # (..., d_in, d_in)
    hessian_dampened: torch.Tensor = dampen_gram(gram=hessian, ratio=dampen_ratio, inplace=False)

    def get_error(w_fake_quant: torch.Tensor, w: torch.Tensor, a_fake_quant: torch.Tensor, a: torch.Tensor) -> tuple[float, ...]:
        w_delta: torch.Tensor = w_fake_quant - w  # (..., d_out, d_in)
        error_qw: float = w_delta.pow(2.).mean().item()
        error_qw_a: float = (a @ w_delta.transpose(-2, -1)).pow(2.).mean().item()
        error_qw_qa: float = (a_fake_quant @ w_fake_quant.transpose(-2, -1) - a @ w.transpose(-2, -1)).pow(2.).mean().item()
        return error_qw, error_qw_a, error_qw_qa

    mxfp4_quant_fn = functools.partial(
        rtn_xxfp4,
        group_size=d,
        scale_dtype=torch.float8_e8m0fnu,
        scale_scale=4.,
        global_scale=1. / 3.,
        fp4_rounding_mode='even',
        high_dtype=None,
        round_dtype=None,
    )
    infinite_int_grid_quant_fn = lambda x, existing_scale=None, **_: {'fake_quant': (x / 1e-1).round() * 1e-1, 'scale_quant': None}

    error_qw_rtn, error_qw_a_rtn, error_qw_qa_rtn = get_error(
        w_fake_quant=rtn_inner(
            x=weight[..., :d],  # (..., d_out, d)
            quant_fn=mxfp4_quant_fn,
        )['coefficient'],  # (..., d_out, d)
        w=weight[..., :d],  # (..., d_out, d)
        a_fake_quant=rtn_inner(
            x=activation[..., :d],  # (..., d_batch, d)
            quant_fn=mxfp4_quant_fn,
        )['coefficient'],  # (..., d_batch, d)
        a=activation[..., :d],  # (..., d_batch, d)
    )
    error_qw_gptq, error_qw_a_gptq, error_qw_qa_gptq = get_error(
        w_fake_quant=gptq_inner(
            x=weight[..., :d],  # (..., d_out, d)
            hessian_sub_space=hessian_dampened[..., :d, :d],  # (..., d, d)
            quant_fn=mxfp4_quant_fn,
        )['coefficient'],  # (..., d_out, d)
        w=weight[..., :d],  # (..., d_out, d)
        a_fake_quant=rtn_inner(
            x=activation[..., :d],  # (..., d_batch, d)
            quant_fn=mxfp4_quant_fn,
        )['coefficient'],  # (..., d_batch, d)
        a=activation[..., :d],  # (..., d_batch, d)
    )
    assert error_qw_rtn <= error_qw_gptq and error_qw_a_rtn >= error_qw_a_gptq and error_qw_qa_rtn >= error_qw_qa_gptq  # verify rtn inner vs gptq inner

    assert babai_inner(
        x=weight[..., :d].flip(dims=(-1,)),  # (..., d_out, d)
        hessian_sub_space=hessian_dampened[..., :d, :d].flip(dims=(-2, -1)),  # (..., d, d)
        quant_fn=mxfp4_quant_fn,
    )['coefficient'].flip(dims=(-1,)).equal(
        gptq_inner(
            x=weight[..., :d],  # (..., d_out, d)
            hessian_sub_space=hessian_dampened[..., :d, :d],  # (..., d, d)
            quant_fn=mxfp4_quant_fn,
        )['coefficient']  # (..., d_out, d)
    )  # verify gptq inner <=> babai inner equivalence

    assert gptq_inner(
        x=weight[..., :d],  # (..., d_out, d)
        hessian_sub_space=hessian_dampened[..., :d, :d],  # (..., d, d)
        quant_fn=mxfp4_quant_fn,
        force_rtn_inner=True,
    )['coefficient'].equal(
        rtn_inner(
            x=weight[..., :d],  # (..., d_out, d)
            quant_fn=mxfp4_quant_fn,
        )['coefficient']  # (..., d_out, d)
    )  # verify gptq inner with force_rtn <=> rtn inner
    assert babai_inner(
        x=weight[..., :d],  # (..., d_out, d)
        hessian_sub_space=hessian_dampened[..., :d, :d],  # (..., d, d)
        quant_fn=mxfp4_quant_fn,
        force_rtn_inner=True,
    )['coefficient'].equal(
        rtn_inner(
            x=weight[..., :d],  # (..., d_out, d)
            quant_fn=mxfp4_quant_fn,
        )['coefficient']  # (..., d_out, d)
    )  # verify babai inner with force_rtn <=> rtn inner

    assert rtn_outer(
        x=weight,  # (..., d_out, d_in)
        hessian=hessian,  # (..., d_in, d_in)
        block_size=d,
        inner_fn=functools.partial(
            transform_inner,
            inner_fn=functools.partial(
                rtn_inner,
                quant_fn=mxfp4_quant_fn,
            ),
        ),
        collate_fn=collate_xxfp4,
    )['coefficient'].equal(
        rtn_inner(
            x=weight,  # (..., d_out, d_in)
            quant_fn=mxfp4_quant_fn,
        )['coefficient']  # (..., d_out, d_in)
    )  # verify rtn outer <=> rtn inner with identity transform

    assert gptq_outer(
        x=weight,  # (..., d_out, d_in)
        hessian=hessian,  # (..., d_in, d_in)
        block_size=d,
        inner_fn=functools.partial(
            transform_inner,
            inner_fn=functools.partial(
                gptq_inner,
                quant_fn=infinite_int_grid_quant_fn,
            ),
        ),
        dampen_ratio=dampen_ratio
    )['coefficient'].equal(
        gptq_inner(
            x=weight,  # (..., d_out, d_in)
            hessian_sub_space=hessian_dampened,  # (..., d_in, d_in)
            quant_fn=infinite_int_grid_quant_fn,
        )['coefficient']  # (..., d_out, d_in)
    )  # verify gptq outer <=> gptq inner with identity transform

    result_tmp_1: dict = gptq_outer(
        x=weight,  # (..., d_out, d_in)
        hessian=hessian,  # (..., d_in, d_in)
        block_size=d,
        inner_fn=functools.partial(
            transform_inner,
            inner_fn=functools.partial(
                gptq_inner,
                quant_fn=mxfp4_quant_fn,
            ),
            get_transform_fn=functools.partial(
                get_transform,
                transform_type='WUSH',
                round_dtype=dtype,
            ),
            use_full_space_hessian=True,
            dampen_ratio=dampen_ratio,
        ),
        collate_fn=collate_xxfp4,
        dampen_ratio=dampen_ratio,
        force_rtn_outer=True,
    )
    result_tmp_2: dict = rtn_outer_parallel(
        x=weight,  # (..., d_out, d_in)
        hessian=hessian,  # (..., d_in, d_in)
        block_size=d,
        inner_fn=functools.partial(
            transform_inner,
            inner_fn=functools.partial(
                gptq_inner,
                quant_fn=mxfp4_quant_fn,
            ),
            get_transform_fn=functools.partial(
                get_transform,
                transform_type='WUSH',
                round_dtype=dtype,
            ),
            use_full_space_hessian=True,
            dampen_ratio=dampen_ratio,
        ),
        collate_fn=collate_parallel_xxfp4,
        dampen_ratio=dampen_ratio,
    )
    assert result_tmp_1['coefficient'].equal(result_tmp_2['coefficient'])
    assert result_tmp_1['e2m1'].view(dtype=torch.uint8).equal(result_tmp_2['e2m1'].view(dtype=torch.uint8))
    assert result_tmp_1['scale_quant'].equal(result_tmp_2['scale_quant'])
    assert result_tmp_1['global_scale'] == result_tmp_2['global_scale']
    assert result_tmp_1['transform'].equal(result_tmp_2['transform'])
    # verify gptq outer force_rtn with wush transform <=> rtn outer with wush transform

    assert babai_outer(
        x=weight,  # (..., d_out, d_in)
        hessian=hessian,  # (..., d_in, d_in)
        block_size=d,
        inner_fn=functools.partial(
            transform_inner,
            inner_fn=functools.partial(
                babai_inner,
                quant_fn=infinite_int_grid_quant_fn,
            ),
        ),
        dampen_ratio=dampen_ratio
    )['coefficient'].equal(
        babai_inner(
            x=weight,  # (..., d_out, d_in)
            hessian_sub_space=hessian_dampened,  # (..., d_in, d_in)
            quant_fn=infinite_int_grid_quant_fn,
        )['coefficient']  # (..., d_out, d_in)
    )  # verify babai outer <=> babai inner with identity transform

    assert babai_outer(
        x=weight,  # (..., d_out, d_in)
        hessian=hessian,  # (..., d_in, d_in)
        block_size=d,
        inner_fn=functools.partial(
            transform_inner,
            inner_fn=functools.partial(
                babai_inner,
                quant_fn=mxfp4_quant_fn,
            ),
            get_transform_fn=functools.partial(
                get_transform,
                transform_type='WUSH',
                round_dtype=dtype,
            ),
            use_full_space_hessian=True,
            dampen_ratio=dampen_ratio,
        ),
        dampen_ratio=dampen_ratio,
        force_rtn_outer=True,
    )['coefficient'].allclose(
        rtn_outer(
            x=weight,  # (..., d_out, d_in)
            hessian=hessian,  # (..., d_in, d_in)
            block_size=d,
            inner_fn=functools.partial(
                transform_inner,
                inner_fn=functools.partial(
                    babai_inner,
                    quant_fn=mxfp4_quant_fn,
                ),
                get_transform_fn=functools.partial(
                    get_transform,
                    transform_type='WUSH',
                    round_dtype=dtype,
                ),
                use_full_space_hessian=True,
                dampen_ratio=dampen_ratio,
            ),
            dampen_ratio=dampen_ratio,
        )['coefficient']  # (..., d_out, d_in)
    )  # verify babai outer force_rtn with wush transform <=> babai outer with wush transform

    for outer_fn, inner_fn in [
        (rtn_outer, rtn_inner),
        (rtn_outer, gptq_inner),
        (gptq_outer, rtn_inner),
        (gptq_outer, gptq_inner),
        # (gptq_outer, babai_inner),
        # (babai_outer, gptq_inner),
        # (babai_outer, babai_inner),
    ]:
        print(f'{outer_fn=}', f'{inner_fn=}')
        for transform_type in 'I', 'R', 'H', 'WUSH':
            for use_full_space_hessian in True, False:
                if not use_full_space_hessian and transform_type != 'WUSH':
                    continue
                result_w: dict = outer_fn(
                    x=weight,  # (..., d_out, d_in)
                    hessian=hessian,  # (..., d_in, d_in)
                    block_size=d,
                    inner_fn=functools.partial(
                        transform_inner,
                        inner_fn=functools.partial(
                            inner_fn,
                            quant_fn=mxfp4_quant_fn,
                        ),
                        get_transform_fn=functools.partial(
                            get_transform,
                            transform_type=transform_type,
                        ),
                        use_full_space_hessian=use_full_space_hessian,
                        dampen_ratio=dampen_ratio,
                    ),
                    collate_fn=collate_xxfp4,
                    dampen_ratio=dampen_ratio,
                )
                a_fake_quant: torch.Tensor = apply_block_transform(
                    x=rtn_inner(
                        x=apply_block_transform(
                            x=activation,  # (..., d_batch, d_in)
                            transform=result_w['transform'],  # (..., d_in // d, d, d)
                            is_inverse_transpose=False,
                            high_dtype=None,
                            round_dtype=None,
                        ),
                        quant_fn=mxfp4_quant_fn,
                    )['coefficient'],  # (..., d_batch, d_in)
                    transform=result_w['transform'].transpose(-2, -1),  # (..., d_in // d, d, d)
                    is_inverse_transpose=True,
                    high_dtype=None,
                    round_dtype=None,
                )  # (..., d_batch, d_in)
                assert dequant_xxfp4(
                    e2m1=result_w['e2m1'],
                    scale_quant=result_w['scale_quant'],
                    global_scale=result_w['global_scale'],
                    dtype=high_dtype,
                    high_dtype=None,
                    round_dtype=None,
                ).equal(result_w['fake_quant'])
                assert apply_block_transform(
                    x=result_w['fake_quant'],
                    transform=result_w['transform'].transpose(-2, -1),
                    is_inverse_transpose=False,
                    high_dtype=None,
                    round_dtype=None,
                ).equal(result_w['coefficient'])
                print(
                    f'{transform_type:<7}',
                    f'{str(use_full_space_hessian):<7}',
                    *[f'{e:.6f}' for e in get_error(w_fake_quant=result_w['coefficient'], w=weight, a_fake_quant=a_fake_quant, a=activation)],
                    sep='\t',
                )
                method_name: str = ''
                if outer_fn == rtn_outer and inner_fn == rtn_inner:
                    method_name: str = 'RTN'
                if outer_fn == gptq_outer and inner_fn == gptq_inner:
                    method_name: str = 'GPTQ'
                if outer_fn == babai_outer and inner_fn == babai_inner:
                    method_name: str = 'BABAI'
                if method_name and transform_type != 'R':
                    r: dict = quantize_layer(
                        x=weight,
                        hessian=hessian,
                        quantize_format_name='MXFP4',
                        method_name=method_name,
                        transform_type=transform_type,
                        quantize_side_name='WA' if use_full_space_hessian else 'W',
                        dampen_ratio=dampen_ratio,
                        round_dtype=None,
                    )
                    assert r['coefficient'].equal(result_w['coefficient'])
        print()

    print('Unit test passed.')


if __name__ == '__main__':
    _unit_test(device=torch.device('cuda'))
