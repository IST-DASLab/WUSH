import torch


def rtn_e2m1(
        x: torch.Tensor,
        mode: str = 'even',
        return_packed: bool = False,
        high_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    round to nearest fp4_e2m1
    x: (..., G)
    mode: str
    return_packed: bool
    high_dtype: torch.dtype | None
    returns: (..., G) fp, (..., G//2) packed (or an empty placeholder tensor if return_packed is False)
    """
    device: torch.device = x.device
    high_dtype = x.dtype if high_dtype is None else high_dtype
    grid: torch.Tensor = torch.as_tensor([-6., -4., -3., -2., -1.5, -1., -.5, -0., 0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=high_dtype, device=device)  # (16) fp
    grid_int: torch.Tensor = torch.as_tensor([-1, -2, -3, -4, -5, -6, -7, -8, 0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.uint8, device=device)  # (16) uint8
    idx: torch.Tensor = torch.bucketize(input=x, boundaries=grid, out_int32=False, right=False)  # (..., G) int64, -inf => 0, inf => 16, nan => 16
    lo, hi = (idx - 1).clamp(min=0, max=15), idx.clamp(min=0, max=15)  # (..., G) int64, (..., G) int64
    value_lo, value_hi = grid[lo], grid[hi]  # (..., G) fp, (..., G) fp
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
    pick_hi: torch.Tensor = (value_hi - x < x - value_lo) | (value_hi - x == x - value_lo) & pick_hi_eq  # (..., G) bool
    idx_picked: torch.Tensor = torch.where(pick_hi, hi, lo)  # (..., G) int64
    zero_pick: torch.Tensor = grid[idx_picked] == 0  # (..., G) bool -- picked a +-0 code
    idx_picked: torch.Tensor = torch.where(zero_pick, torch.where(x.signbit(), 7, 8), idx_picked)  # (..., G) int64
    y: torch.Tensor = grid[idx_picked]  # (..., G) fp
    if not return_packed:
        return y, torch.empty(0, dtype=torch.float4_e2m1fn_x2, device=device)  # (..., G) fp, (0)
    assert y.size(-1) % 2 == 0
    y_int: torch.Tensor = grid_int[idx_picked]  # (..., G) uint8
    y_int_packed: torch.Tensor = ((y_int[..., 1::2] & 0xF) << 4 | y_int[..., ::2] & 0xF).view(dtype=torch.float4_e2m1fn_x2)  # (..., G//2)
    return y, y_int_packed  # (..., G) fp, (... G//2)


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
    x_int: torch.Tensor = torch.stack([x_int_packed & 0xF, (x_int_packed >> 4) & 0xF], dim=-1).flatten(start_dim=-2)  # (..., G) uint8
    x: torch.Tensor = grid[x_int.to(dtype=torch.int64)]  # (..., G) fp
    return x  # (..., G) fp


def rtn_xxfp4(
        x: torch.Tensor,
        group_size: int = -1,
        scale_dtype: torch.dtype = torch.float64,
        scale_scale: torch.Tensor | float = 6.,
        global_scale: torch.Tensor | float = 1.,
        fp4_rounding_mode: str = 'even',
        existing_scale: torch.Tensor | None = None,
        high_dtype: torch.dtype | None = None,
        round_dtype: torch.dtype | None = None,
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
    high_dtype: torch.dtype | None
    round_dtype: torch.dtype | None
    """
    dtype, device = x.dtype, x.device
    high_dtype = dtype if high_dtype is None else high_dtype
    round_dtype = dtype if round_dtype is None else round_dtype
    group_size = x.size(-1) if group_size <= 0 else group_size
    # gs / ss may be a python scalar OR a single-element tensor. Normalize gs to a high_dtype tensor so x_scaled
    # and the dequant are tensor ops (IEEE, full precision); ss stays as-is -- (ss * gs) is already a tensor (gs
    # is), so the scale divide is IEEE too. Keep the python gs value for the result (no device->host .item()).
    global_scale_value: float = global_scale.item() if isinstance(global_scale, torch.Tensor) else global_scale
    global_scale = torch.as_tensor(global_scale, dtype=high_dtype, device=device)

    x_reshaped: torch.Tensor = x.to(dtype=high_dtype).unflatten(dim=-1, sizes=(-1, group_size))  # (..., C//G, G)

    if existing_scale is None:
        x_reshaped_abs_max: torch.Tensor = x_reshaped.abs().amax(dim=-1, keepdim=True)  # (..., C//G, 1)
        scale: torch.Tensor = x_reshaped_abs_max / (scale_scale * global_scale)  # (..., C//G, 1); gs folded into the divisor
        scale_quantized: torch.Tensor = scale.clamp(min=torch.finfo(scale_dtype).min, max=torch.finfo(scale_dtype).max).to(dtype=scale_dtype)  # (..., C//G, 1), clamp to the representable range (e8m0 min > 0 also prevents zero scales; e4m3 can still produce zero scales for all-zero groups, which dequantize consistently to zero)
    else:
        scale_quantized: torch.Tensor = existing_scale[..., None]  # (..., C//G, 1)

    scale_dequantized: torch.Tensor = scale_quantized.to(dtype=high_dtype)  # (..., C//G, 1)

    # x_reshaped_scaled: torch.Tensor = x_reshaped * scale_scale / x_reshaped_abs_max  # (..., C//G, G)
    x_reshaped_scaled: torch.Tensor = x_reshaped / (scale_dequantized * global_scale)  # (..., C//G, G)
    x_reshaped_scaled_dequantized, x_reshaped_scaled_quantized_packed = rtn_e2m1(x=x_reshaped_scaled, mode=fp4_rounding_mode, return_packed=True, high_dtype=high_dtype)  # (..., C//G, G), (..., C//G, G//2)
    x_reshaped_dequantized: torch.Tensor = x_reshaped_scaled_dequantized * (scale_dequantized * global_scale)  # (..., C//G, G)
    x_dequantized: torch.Tensor = x_reshaped_dequantized.flatten(start_dim=-2)  # (..., C)

    result: dict = {
        'fake_quant': x_dequantized.to(dtype=round_dtype).to(dtype=dtype),  # (..., C)
        'e2m1': x_reshaped_scaled_quantized_packed.flatten(start_dim=-2),  # (..., C//2)
        'scale_quant': scale_quantized[..., 0],  # (..., C//G)
        'global_scale': global_scale_value,  # python float (captured pre-conversion; no device->host sync)
    }
    return result


def dequant_xxfp4(
        e2m1: torch.Tensor,
        scale_quant: torch.Tensor,
        global_scale: torch.Tensor | float = 1.,
        dtype: torch.dtype = torch.float64,
        high_dtype: torch.dtype | None = None,
        round_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    dequantize xxfp4 group format
    e2m1: (..., C//2) float4_e2m1fn_x2
    scale_quant: (..., C//G)
    global_scale: float
    dtype: torch.dtype
    high_dtype: torch.dtype | None
    round_dtype: torch.dtype | None
    returns: (..., C)
    """
    high_dtype = dtype if high_dtype is None else high_dtype
    round_dtype = dtype if round_dtype is None else round_dtype
    global_scale = torch.as_tensor(global_scale, dtype=high_dtype, device=scale_quant.device)  # scalar or single-element tensor -> high_dtype tensor
    scale_quantized: torch.Tensor = scale_quant[..., None]  # (..., C//G, 1)
    x_reshaped_scaled_quantized_packed: torch.Tensor = e2m1.unflatten(dim=-1, sizes=(scale_quantized.size(-2), -1))  # (..., C//G, G//2)
    x_reshaped_scaled_dequantized: torch.Tensor = unpack_e2m1(x_packed=x_reshaped_scaled_quantized_packed, dtype=high_dtype)  # (..., C//G, G)
    scale_dequantized: torch.Tensor = scale_quantized.to(dtype=high_dtype)  # (..., C//G, 1)
    x_reshaped_dequantized: torch.Tensor = x_reshaped_scaled_dequantized * (scale_dequantized * global_scale)  # (..., C//G, G)
    x_dequantized: torch.Tensor = x_reshaped_dequantized.flatten(start_dim=-2)  # (..., C)
    return x_dequantized.to(dtype=round_dtype).to(dtype=dtype)  # (..., C)


def _unit_test(
        device: torch.device = torch.device('cuda'),
) -> None:
    """
    unit test
    """
    torch.manual_seed(seed=0)
    dtype, high_dtype = torch.bfloat16, torch.float64

    def mxfp4_reference(
            x: torch.Tensor,
    ) -> torch.Tensor:
        """
        https://arxiv.org/abs/2502.20586
        x: (..., C)
        """
        dtype: torch.dtype = x.dtype
        group_size: int = 32
        x_reshaped: torch.Tensor = x.to(dtype=high_dtype).unflatten(dim=-1, sizes=(-1, group_size))  # (..., C//G, G)
        x_reshaped_abs_max: torch.Tensor = x_reshaped.abs().amax(dim=-1, keepdim=True)  # (..., C//G, 1)
        scale: torch.Tensor = (x_reshaped_abs_max.log2().floor() - 2.).exp2()  # (..., C//G, 1)
        scale_quantized: torch.Tensor = scale.to(dtype=torch.float8_e8m0fnu)  # (..., C//G, 1)
        scale_dequantized: torch.Tensor = scale_quantized.to(dtype=high_dtype)  # (..., C//G, 1)
        x_reshaped_scaled: torch.Tensor = x_reshaped * 3. / 4. / scale_dequantized  # (..., C//G, G)
        x_reshaped_scaled_dequantized, x_reshaped_scaled_quantized_packed = rtn_e2m1(x_reshaped_scaled, mode='zero', return_packed=True)  # (..., C//G, G), (..., C//G, G//2)
        x_reshaped_dequantized: torch.Tensor = x_reshaped_scaled_dequantized * scale_dequantized * 4. / 3.  # (..., C//G, G)
        x_dequantized: torch.Tensor = x_reshaped_dequantized.flatten(start_dim=-2)  # (..., C)
        result: dict = {
            'fake_quant': x_dequantized.to(dtype=dtype),  # (..., C)
            'e2m1': x_reshaped_scaled_quantized_packed.flatten(start_dim=-2),  # (..., C//2)
            'scale_quant': scale_quantized[..., 0],  # (..., C//G)
        }
        return result['fake_quant']

    def get_and_verify_fake_quant(result: dict) -> torch.Tensor:
        fake_quant: torch.Tensor = dequant_xxfp4(
            e2m1=result['e2m1'],
            scale_quant=result['scale_quant'],
            global_scale=result['global_scale'],
            dtype=dtype,
            high_dtype=high_dtype,
            round_dtype=high_dtype,
        )
        assert fake_quant.equal(result['fake_quant'])
        return fake_quant

    x: torch.Tensor = torch.randn(1024, 512, dtype=dtype, device=device) * 1e-1
    result_mxfp4: dict = rtn_xxfp4(
        x=x,
        group_size=32,
        scale_dtype=torch.float8_e8m0fnu,
        scale_scale=4.,
        global_scale=1. / 3.,
        fp4_rounding_mode='zero',
        high_dtype=high_dtype,
        round_dtype=dtype,
    )
    x0_mxfp4: torch.Tensor = get_and_verify_fake_quant(result=result_mxfp4)
    x1_mxfp4: torch.Tensor = mxfp4_reference(x=x)
    assert x1_mxfp4.equal(x0_mxfp4)

    rtn_xxfp4(
        x=x,
        group_size=16,
        scale_dtype=torch.float8_e4m3fn,
        scale_scale=6.,
        global_scale=.1,
        fp4_rounding_mode='zero',
        high_dtype=high_dtype,
        round_dtype=dtype,
    )  # result_nvfp4

    print('Unit test passed.')


if __name__ == '__main__':
    _unit_test(device=torch.device('cuda'))
