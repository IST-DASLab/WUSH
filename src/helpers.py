from typing import Tuple

import torch


def decompose_dim(size: int) -> Tuple[int, int]:
    a = int(size ** 0.5)
    if a ** 2 == size:
        return a, a
    for i in range(a, 0, -1):
        if size % i == 0:
            return i, size // i

def split_dim(x: torch.Tensor, num_splits: int, dim: int = -1) -> torch.Tensor:
    if dim == -1:
        dim = x.ndim - 1
    new_shape = (*x.shape[:dim], num_splits, x.shape[dim] // num_splits, *x.shape[dim+1:])
    return x.reshape(new_shape)
