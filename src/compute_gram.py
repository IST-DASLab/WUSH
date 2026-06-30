import functools

import torch


def get_gram(
        x: torch.Tensor,
        dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    gram = x.t() @ x

    Inputs:
      x: (..., m, d_in)
      dtype: torch.dtype | None

    Returns:
      gram: (..., d_in, d_in)
    """

    dtype = x.dtype if dtype is None else dtype
    x = x.to(dtype=dtype)  # (..., m, d_in)
    return torch.einsum('...mi,...mj->...ij', x, x)  # (..., d_in, d_in)


def get_diag_block_gram(
        x: torch.Tensor,
        size: int | None = None,
        dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    gram = x.t() @ x, blockwise

    Inputs:
      x: (..., m, d_in)
      size: int | None, d
      dtype: torch.dtype | None

    Returns:
      gram: (..., d_in // d, d, d)
    """

    dtype = x.dtype if dtype is None else dtype
    size = x.size(-1) if size is None or size <= 0 else size
    x = x.unflatten(dim=-1, sizes=(-1, size)).to(dtype=dtype)  # (..., m, d_in // d, d)
    return torch.einsum('...mgi,...mgj->...gij', x, x)  # (..., d_in // d, d, d)


def get_diag_block(
        matrix: torch.Tensor,
        size: int | None = None,
) -> torch.Tensor:
    """
    Fetch the diagonal blocks of the Gram matrix

    Inputs:
      matrix: (..., d_in, d_in)
      size: int | None, d

    Returns:
      matrix_diag_block: (..., d_in // d, d, d), view
    """

    size = matrix.size(-1) if size is None or size <= 0 else size
    return matrix.unflatten(
        dim=-2, sizes=(-1, size),
    ).unflatten(
        dim=-1, sizes=(-1, size),
    ).diagonal(
        offset=0, dim1=-4, dim2=-2,
    ).movedim(
        source=-1, destination=-3,
    )  # (..., d_in // d, d, d)


def diag_embed_block(
        matrix_diag_block: torch.Tensor,
        out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    torch.diag_embed squared block version

    Inputs:
      matrix_diag_block: (..., d_in // d, d, d)
      out: (..., d_in, d_in), inplace

    Returns:
      out: (..., d_in, d_in), inplace
    """

    *batch_dims, n_blocks, _, d = matrix_diag_block.shape
    d_in: int = n_blocks * d
    out = torch.zeros(*batch_dims, d_in, d_in, dtype=matrix_diag_block.dtype, device=matrix_diag_block.device) if out is None else out
    get_diag_block(matrix=out, size=d).copy_(matrix_diag_block)
    return out


def dampen_gram(
        gram: torch.Tensor,
        ratio: float = 0.,
        inplace: bool = False,
) -> torch.Tensor:
    """
    Dampen the diagonal of the Gram matrix
    gram = gram + ratio * tr(gram) * I / n

    Inputs:
      gram: (..., n, n)
      ratio: float
      inplace: bool

    Returns:
      gram: (..., n, n)
    """

    if not inplace:
        gram = gram.clone()  # (..., n, n)
    gram.diagonal(offset=0, dim1=-2, dim2=-1).add_(ratio * gram.diagonal(offset=0, dim1=-2, dim2=-1).mean(dim=-1, keepdim=True))  # (..., n) <= (..., 1)
    return gram  # (..., n, n)


def invert_gram(
        gram: torch.Tensor,
        inplace: bool = False,
) -> torch.Tensor:
    cholesky_lower :torch.Tensor = torch.linalg.cholesky_ex(
        gram,
        upper=False,
        check_errors=True,
        out=(gram, torch.empty(*gram.shape[:-2], dtype=torch.int32, device=gram.device)) if inplace else None,
    ).L  # (..., n, n)
    return torch.cholesky_inverse(cholesky_lower, upper=False, out=cholesky_lower)  # (..., n, n)


def cholesky_decompose_gram(
        gram: torch.Tensor,
        sizes: tuple[int, ...] | list[int] | None = None,
        left_upper: bool = False,
        return_transposed: bool = False,
) -> list[torch.Tensor]:
    """
    Decompose the Gram matrix
    gram = L1 @ L2 @ ... @ Lk @ Lk.t() @ ... @ L2.t() @ L1.t() = U1 @ U2 @ ... @ Uk @ Uk.t() @ ... @ U2.t() @ U1.t() (L: lower triangular, U: upper triangular)

    Inputs:
      gram: (..., n, n)
      sizes: list[int], descending block sizes
      return_transposed: bool, whether to return L.t() or U.t()

    Returns:
      result: list of (..., n // size_i, size_i, size_i), one per size_i in sizes
    """

    sizes = (gram.size(-1),) if sizes is None else sizes

    if left_upper:
        gram_flip: torch.Tensor = gram.flip(dims=(-2, -1))  # (..., n, n)
        info: torch.Tensor = torch.empty(*gram.shape[:-2], dtype=torch.int32, device=gram.device)  # (...)
        triangles: torch.Tensor = torch.linalg.cholesky_ex(gram_flip, upper=return_transposed, check_errors=True, out=(gram_flip, info)).L.flip(dims=(-2, -1))  # (..., n, n)
    else:
        triangles: torch.Tensor = torch.linalg.cholesky_ex(gram, upper=return_transposed, check_errors=True).L  # (..., n, n)

    triangles: torch.Tensor = triangles[..., None, :, :]  # (..., 1, n, n)
    results: list[torch.Tensor] = []
    for size in sizes[1:]:
        small_triangles: torch.Tensor = get_diag_block(matrix=triangles, size=size)  # (..., a // b, b, b)
        if return_transposed:
            large_triangles: torch.Tensor = torch.linalg.solve_triangular(
                small_triangles,  # (..., a // b, b, b)
                triangles.unflatten(dim=-2, sizes=(-1, size)),  # (..., a // b, b, a)
                upper=not left_upper,
                left=True,
                unitriangular=False,
            ).flatten(start_dim=-3, end_dim=-2)  # (..., a, a)
        else:
            large_triangles: torch.Tensor = torch.linalg.solve_triangular(
                small_triangles,  # (..., a // b, b, b)
                triangles.unflatten(dim=-1, sizes=(-1, size)).transpose(-3, -2),  # (..., a // b, a, b)
                upper=left_upper,
                left=False,
                unitriangular=False,
            ).transpose(-3, -2).flatten(start_dim=-2, end_dim=-1)  # (..., a, a)
        results.append(large_triangles)
        triangles: torch.Tensor = small_triangles
    results.append(triangles)

    results: list[torch.Tensor] = [r.flatten(start_dim=gram.dim() - 2, end_dim=-3) for r in results]
    return results  # list of (..., n // size_i, size_i, size_i)


def _unit_test(device: torch.device = torch.device('cuda')):
    """
    Unit test

    Inputs:
      device: torch.device
    """

    torch.manual_seed(seed=0)

    dtype, high_dtype = torch.bfloat16, torch.float64
    atol, rtol = 1e-8, 1e-6
    batch_dims: tuple[int, ...] = 2, 3
    m, d_in, d = 31, 16, 8

    x: torch.Tensor = torch.randn(*batch_dims, m, d_in, dtype=dtype, device=device)  # (..., m, d_in)

    assert get_gram(x=x, dtype=high_dtype).equal(x.to(dtype=high_dtype).transpose(-2, -1) @ x.to(dtype=high_dtype))
    assert get_diag_block_gram(x=x, size=d, dtype=high_dtype).equal(get_gram(x=x.unflatten(dim=-1, sizes=(-1, d)).transpose(-3, -2), dtype=high_dtype))
    assert get_diag_block(matrix=get_gram(x=x, dtype=high_dtype), size=d).equal(get_diag_block_gram(x=x, size=d, dtype=high_dtype))
    assert get_diag_block(matrix=diag_embed_block(matrix_diag_block=get_diag_block_gram(x=x, size=d, dtype=high_dtype), out=None), size=d).equal(get_diag_block_gram(x=x, size=d, dtype=high_dtype))

    damp_ratio: float = 1e-2
    gram: torch.Tensor = get_gram(x=x, dtype=high_dtype)  # (..., d_in, d_in)
    dampen_gram(gram=gram, ratio=damp_ratio, inplace=True)
    assert gram.equal(get_gram(x=x, dtype=high_dtype) + damp_ratio * get_gram(x=x, dtype=high_dtype).diagonal(offset=0, dim1=-2, dim2=-1).mean(dim=-1)[..., None, None] * torch.eye(d_in, dtype=high_dtype, device=device))
    assert (invert_gram(gram=gram, inplace=False) @ gram).allclose(torch.eye(d_in, dtype=high_dtype, device=device), atol=atol, rtol=rtol)

    for sizes in [d_in, d_in // 2, 1], [d_in, d_in // 2], [d_in, 1], [d_in]:
        for left_upper in False, True:
            for return_transposed in False, True:
                results: list[torch.Tensor] = cholesky_decompose_gram(gram=gram, sizes=sizes, left_upper=left_upper, return_transposed=return_transposed)
                assert len(results) == len(sizes)
                for i, (r, s) in enumerate(zip(results, sizes)):
                    assert r.size(-2) == r.size(-1) == s
                    if i < len(sizes) - 1:
                        assert (get_diag_block(matrix=r, size=sizes[i+1]) == torch.eye(sizes[i+1], dtype=high_dtype, device=device)).all()
                    if left_upper ^ return_transposed:
                        assert r.triu().equal(r)
                    else:
                        assert r.tril().equal(r)
                m = functools.reduce(torch.matmul, [diag_embed_block(r, out = None) for r in (results[::-1] if return_transposed else results)])
                assert get_gram(m if return_transposed else m.transpose(-2, -1), dtype=high_dtype).allclose(gram, atol=atol, rtol=rtol)

    print('Unit test passed.')


if __name__ == '__main__':
    _unit_test(device=torch.device('cuda'))
