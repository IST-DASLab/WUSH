import torch

try:
    # Force the scipy fallback for now. Remove this `raise` to switch to the `fast_hadamard_transform` package.
    raise ImportError
    from fast_hadamard_transform import hadamard_transform
    get_normalized_hadamard_transform = lambda size, dtype=torch.float64, device=torch.device('cuda'): hadamard_transform(torch.eye(size, dtype=dtype, device=device), scale=size ** -.5)
except ImportError:
    import scipy
    get_normalized_hadamard_transform = lambda size, dtype=torch.float64, device=torch.device('cpu'): torch.as_tensor(scipy.linalg.hadamard(size), dtype=dtype, device=device) * size ** -.5


def apply_block_transform(
        x: torch.Tensor,
        transform: torch.Tensor | None,
        is_inverse_transpose: bool = False,
        high_dtype: torch.dtype | None = None,
        round_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    x_transformed = x @ transform.t() or x_transformed = x @ transform.inverse()

    Inputs:
      x: (..., m, d_in)
      transform: (..., d_in // d, d, d)
      is_inverse_transpose: bool, do not use if you want to save GPU memory; requires high_dtype to be fp32/fp64 (uses linalg solve)
      high_dtype: torch.dtype | None
      round_dtype: torch.dtype | None

    Returns:
      x_transformed: (..., m, d_in)
    """

    if transform is None:
        return x  # (..., m, d_in)

    dtype: torch.dtype = x.dtype
    high_dtype = dtype if high_dtype is None else high_dtype
    round_dtype = dtype if round_dtype is None else round_dtype
    size: int = transform.size(-1)
    x, transform = x.to(dtype=high_dtype), transform.to(dtype=high_dtype)
    if is_inverse_transpose:
        # x_transformed: torch.Tensor = torch.linalg.solve_ex(
        #     transform[..., None, :, :, :],  # (..., 1, d_in // d, d, d)
        #     x.unflatten(dim=-1, sizes=(-1, 1, size)),  # (..., m, d_in // d, 1, d)
        #     left=False,
        #     check_errors=True,
        # ).result.flatten(start_dim=-3)  # (..., m, d_in), without transpose but with broadcast
        x_transformed: torch.Tensor = torch.linalg.solve_ex(
            transform,  # (..., d_in // d, d, d)
            x.unflatten(dim=-1, sizes=(-1, size)).transpose(-3, -2),  # (..., d_in // d, m, d)
            left=False,
            check_errors=True,
        ).result.transpose(-3, -2).flatten(start_dim=-2)  # (..., m, d_in), without broadcast but with transpose
    else:
        x_transformed: torch.Tensor = torch.einsum(
            '...gji,...mgi->...mgj',
            transform,  # (..., d_in // d, d, d)
            x.unflatten(dim=-1, sizes=(-1, size)),  # (..., m, d_in // d, d)
        ).flatten(start_dim=-2)  # (..., m, d_in)
    return x_transformed.to(dtype=round_dtype, memory_format=torch.contiguous_format).to(dtype=dtype)  # (..., m, d_in)


def apply_block_transform_gram(
        gram: torch.Tensor,
        transform: torch.Tensor | None,
        is_inverse_transpose: bool = False,
) -> torch.Tensor:
    """
    gram_transformed = transform @ gram @ transform.t() or gram_transformed = transform.t().inverse() @ gram @ transform.inverse()

    Inputs:
      gram: (..., d_in, d_in)
      transform: (..., d_in // d, d, d)
      is_inverse_transpose: bool, do not use if you want to save GPU memory

    Returns:
      gram_transformed: (..., d_in, d_in)
    """

    dtype: torch.dtype = gram.dtype
    return apply_block_transform(
        x=apply_block_transform(
            x=gram,  # (..., d_in, d_in)
            transform=transform,  # (..., d_in // d, d, d)
            is_inverse_transpose=is_inverse_transpose,
            high_dtype=dtype,
            round_dtype=dtype,
        ).transpose(-2, -1),  # (..., d_in, d_in)
        transform=transform,  # (..., d_in // d, d, d)
        is_inverse_transpose=is_inverse_transpose,
        high_dtype=dtype,
        round_dtype=dtype,
    )  # (..., d_in, d_in)


def get_inversed_transposed_transform(
        transform: torch.Tensor,
        high_dtype: torch.dtype | None = None,
        round_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    transform_t_inv = transform.t().inverse()

    Inputs:
      transform: (..., d, d)
      high_dtype: torch.dtype | None
      round_dtype: torch.dtype | None

    Returns:
      transform_t_inv: (..., d, d)
    """

    dtype: torch.dtype = transform.dtype
    high_dtype = dtype if high_dtype is None else high_dtype
    round_dtype = dtype if round_dtype is None else round_dtype
    transform_t_inv: torch.Tensor = torch.linalg.inv_ex(transform.transpose(-2, -1).to(dtype=high_dtype), check_errors=True).inverse  # (..., d, d)

    transform_t_inv: torch.Tensor = transform_t_inv.to(dtype=round_dtype, memory_format=torch.contiguous_format).to(dtype=dtype)  # (..., d, d)
    assert transform_t_inv.isfinite().all()
    return transform_t_inv  # (..., d, d)


def get_random_orthogonal_transform(
        *batch_dims,
        size: int,
        dtype: torch.dtype,
        device: torch.device,
        enforce_rotation: bool = False,
        high_dtype: torch.dtype | None = None,
        round_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Compute the random orthogonal transform

    Inputs:
      batch_dims:
      size: int
      dtype: torch.dtype | None
      device: torch.device
      enforce_rotation: bool
      high_dtype: torch.dtype | None
      round_dtype: torch.dtype | None

    Returns:
      transform: (..., d, d)
    """

    high_dtype = dtype if high_dtype is None else high_dtype
    round_dtype = dtype if round_dtype is None else round_dtype
    q, r = torch.linalg.qr(torch.randn(*batch_dims, size, size, dtype=high_dtype, device=device), mode='reduced')  # (..., d, d)
    transform: torch.Tensor = torch.einsum(
        '...ij,...j->...ij',
        q,  # (..., d, d)
        r.diagonal(offset=0, dim1=-2, dim2=-1).sgn(),  # (..., d)
    )  # (..., d, d), Mezzadri sign correction, uniform (Haar) on O(d)
    if enforce_rotation:
        transform: torch.Tensor = torch.cat([
            transform[..., :1] * torch.linalg.det(transform)[..., None, None],  # (..., d, 1)
            transform[..., 1:],  # (..., d, d - 1)
        ], dim=-1)  # (..., d, d), uniform on SO(d)

    transform: torch.Tensor = transform.to(dtype=round_dtype, memory_format=torch.contiguous_format).to(dtype=dtype)  # (..., d, d)
    assert transform.isfinite().all()
    return transform  # (..., d, d)


def get_wush_transform(
        gram_weight: torch.Tensor,
        gram_activation: torch.Tensor,
        preserve_norm: str = 'balanced',
        use_hadamard: bool = True,
        high_dtype: torch.dtype | None = None,
        round_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Compute the WUSH transform

    Inputs:
      gram_weight: (..., d, d)
      gram_activation: (..., d, d)
      preserve_norm: str, 'balanced', 'weight', 'activation'
      use_hadamard: bool, whether to use Hadamard (WUSH) or not (WUS)
      high_dtype: torch.dtype | None
      round_dtype: torch.dtype | None

    Returns:
      transform: (..., d, d)
    """

    device = gram_weight.device
    dtype: torch.dtype = gram_weight.dtype
    high_dtype = dtype if high_dtype is None else high_dtype
    round_dtype = dtype if round_dtype is None else round_dtype
    size: int = gram_weight.size(-1)

    gram_weight, gram_activation = gram_weight.to(dtype=high_dtype), gram_activation.to(dtype=high_dtype)  # (..., d, d), (..., d, d)
    cholesky_weight: torch.Tensor = torch.linalg.cholesky_ex(gram_weight, upper=False, check_errors=True).L  # (..., d, d), lower triangular, cholesky_weight @ cholesky_weight.t() = gram_weight
    gram: torch.Tensor = torch.einsum(
        '...ai,...ab,...bj->...ij',
        cholesky_weight,  # (..., d, d)
        gram_activation,  # (..., d, d)
        cholesky_weight,  # (..., d, d)
    )  # (..., d, d), cholesky_weight.t() @ gram_activation @ cholesky_weight
    # eigh of W'.t() @ M_X @ W' = (W'.t() @ X')(W'.t() @ X').t() = U @ S^2 @ U.t() recovers the U and S of SVD(W'.t() @ X') without forming X';
    # sign/ordering differences of U only permute/flip Hadamard columns, which keeps the transform optimal but not bitwise equal to a literal SVD implementation
    eigenvalues, eigenvectors = torch.linalg.eigh(gram, UPLO='L')  # (..., d) ascending, (..., d, d)

    match preserve_norm:
        case 'balanced':
            scaling: torch.Tensor = torch.ones((), dtype=high_dtype, device=device)  # ()
        case 'weight':
            scaling: torch.Tensor = ((eigenvalues ** .5).mean(dim=-1) / gram_weight.diagonal(offset=0, dim1=-2, dim2=-1).mean(dim=-1)) ** .5  # (...)
        case 'activation':
            scaling: torch.Tensor = (gram_activation.diagonal(offset=0, dim1=-2, dim2=-1).mean(dim=-1) / (eigenvalues ** .5).mean(dim=-1)) ** .5  # (...)
        case _:
            raise NotImplementedError

    if use_hadamard:
        hadamard: torch.Tensor = get_normalized_hadamard_transform(size, dtype=high_dtype, device=device)  # (d, d)
    else:
        hadamard: torch.Tensor = torch.eye(size, dtype=high_dtype, device=device)  # (d, d)

    transform: torch.Tensor = torch.einsum(
        '...,...is,...s,...ks,...jk->...ij',
        scaling,  # (...)
        hadamard,  # (d, d)
        eigenvalues ** -.25,  # (..., d)
        eigenvectors,  # (..., d, d)
        cholesky_weight,  # (..., d, d)
    )  # (..., d, d), scaling * hadamard @ (eigenvalues ** -.25).diag() @ eigenvectors.t() @ cholesky_weight.t()

    transform: torch.Tensor = transform.to(dtype=round_dtype, memory_format=torch.contiguous_format).to(dtype=dtype)  # (..., d, d)
    assert transform.isfinite().all()
    return transform  # (..., d, d)


def _unit_test(
        device: torch.device = torch.device('cuda'),
) -> None:
    """
    Unit test

    Inputs:
      device: torch.device
    """

    from compute_gram import get_diag_block_gram, get_gram

    torch.manual_seed(seed=0)
    dtype, high_dtype = torch.float32, torch.float64
    atol, rtol = 1e-4, 1e-3

    def is_power_of_two(d: int) -> bool:
        return d > 0 and (d & (d - 1)) == 0

    def get_variance(
            x: torch.Tensor,  # (..., m, d_in)
    ) -> torch.Tensor:
        return x.to(dtype=high_dtype).pow(2.).mean(dim=-2).unflatten(dim=-1, sizes=(-1, d))  # (..., d_in // d, d)

    batch_dims: tuple[int, ...] = 3, 5
    d: int = 8
    d_batch, d_in, d_out = 61, d * 7, 67
    assert is_power_of_two(d) and d_in % d == 0 and d <= min(d_batch, d_out)

    transform_r: torch.Tensor = get_random_orthogonal_transform(
        *batch_dims,
        size=d,
        dtype=high_dtype,
        device=device,
        enforce_rotation=True,  # only for testing
        high_dtype=high_dtype,
        round_dtype=high_dtype,  # only for testing
    )
    assert get_gram(x=transform_r, dtype=high_dtype).allclose(torch.eye(d, dtype=high_dtype, device=device), atol=atol, rtol=rtol)  # test if is orthogonal
    assert torch.linalg.det(transform_r).allclose(torch.ones((), dtype=high_dtype, device=device), atol=atol, rtol=rtol)  # test if is rotation (no reflection)

    weight: torch.Tensor = torch.randn(*batch_dims, d_out, d_in, dtype=dtype, device=device)  # (..., d_out, d_in)
    activation: torch.Tensor = torch.randn(*batch_dims, d_batch, d_in, dtype=dtype, device=device)  # (..., d_batch, d_in)

    gram_weight: torch.Tensor = get_diag_block_gram(x=weight, size=d, dtype=high_dtype) / d_out  # (..., d_in // d, d, d)
    gram_activation: torch.Tensor = get_diag_block_gram(x=activation, size=d, dtype=high_dtype) / d_batch  # (..., d_in // d, d, d)
    variance_weight: torch.Tensor = get_variance(weight)  # (..., d_in // d, d)
    variance_activation: torch.Tensor = get_variance(activation)  # (..., d_in // d, d)

    for preserve_norm in 'balanced', 'weight', 'activation':
        transform_wush: torch.Tensor = get_wush_transform(
            gram_weight=gram_weight,
            gram_activation=gram_activation,
            preserve_norm=preserve_norm,
            use_hadamard=True,
            high_dtype=high_dtype,
            round_dtype=high_dtype,  # only for testing
        )  # (..., d_in // d, d, d)
        transform_xvsh: torch.Tensor = get_inversed_transposed_transform(transform=transform_wush, high_dtype=high_dtype, round_dtype=high_dtype)  # (..., d_in // d, d, d)

        weight_transformed: torch.Tensor = apply_block_transform(x=weight, transform=transform_wush, is_inverse_transpose=True, high_dtype=high_dtype, round_dtype=high_dtype)  # (..., d_out, d_in)
        weight_transformed_: torch.Tensor = apply_block_transform(x=weight, transform=transform_xvsh, is_inverse_transpose=False, high_dtype=high_dtype, round_dtype=high_dtype)  # (..., d_out, d_in)
        assert weight_transformed_.allclose(weight_transformed, atol=atol, rtol=rtol)  # test of the inverse transpose function
        activation_transformed: torch.Tensor = apply_block_transform(x=activation, transform=transform_wush, is_inverse_transpose=False, high_dtype=high_dtype, round_dtype=high_dtype)  # (..., d_batch, d_in)
        assert apply_block_transform_gram(gram=get_gram(x=weight, dtype=high_dtype), transform=transform_wush, is_inverse_transpose=True).allclose(get_gram(x=weight_transformed, dtype=high_dtype), atol=atol, rtol=rtol)  # test of the gram transform function
        assert apply_block_transform_gram(gram=get_gram(x=activation, dtype=high_dtype), transform=transform_wush, is_inverse_transpose=False).allclose(get_gram(x=activation_transformed, dtype=high_dtype), atol=atol, rtol=rtol)  # test of the gram transform function

        gram_weight_transformed: torch.Tensor = get_diag_block_gram(x=weight_transformed, size=d, dtype=high_dtype) / d_out  # (..., d_in // d, d, d)
        gram_activation_transformed: torch.Tensor = get_diag_block_gram(x=activation_transformed, size=d, dtype=high_dtype) / d_batch  # (..., d_in // d, d, d)
        ratio: torch.Tensor = gram_weight_transformed / gram_activation_transformed  # (..., d_in // d, d, d)
        assert ratio.allclose(ratio.mean(dim=(-2, -1), keepdim=True), atol=atol, rtol=rtol)  # test if weight and activation grams are proportional after transform
        abs_eigenvectors: torch.Tensor = torch.linalg.eigh(gram_weight_transformed, UPLO='L').eigenvectors.abs()  # (..., d_in // d, d, d)
        assert abs_eigenvectors.allclose(abs_eigenvectors.mean(), atol=atol, rtol=rtol)  # test if the gram eigenbasis is Hadamard after transform

        variance_weight_transformed: torch.Tensor = get_variance(weight_transformed)  # (..., d_in // d, d)
        variance_activation_transformed: torch.Tensor = get_variance(activation_transformed)  # (..., d_in // d, d)
        assert variance_weight_transformed.allclose(variance_weight_transformed.mean(dim=-1, keepdim=True), atol=atol, rtol=rtol)  # test if weight marginal variances are the same after transform
        assert variance_activation_transformed.allclose(variance_activation_transformed.mean(dim=-1, keepdim=True), atol=atol, rtol=rtol)  # test if activation marginal variances are the same after transform
        match preserve_norm:  # test if norms are preserved in the specified way before and after transform
            case 'balanced':
                assert variance_weight_transformed.mean(dim=-1).allclose(variance_activation_transformed.mean(dim=-1), atol=atol, rtol=rtol)
            case 'weight':
                assert variance_weight_transformed.mean(dim=-1).allclose(variance_weight.mean(dim=-1), atol=atol, rtol=rtol)
            case 'activation':
                assert variance_activation_transformed.mean(dim=-1).allclose(variance_activation.mean(dim=-1), atol=atol, rtol=rtol)

    print('Unit test passed.')


if __name__ == '__main__':
    _unit_test(device=torch.device('cuda'))
