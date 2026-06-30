#
# Licensed under the Apache License, Version 2.0 (the "License").
#
"""torch.compile-traceable mirror of the qutlass op interface, in the separate `qutlass_ops::` namespace.

qutlass's Python entry points allocate outputs in Python and then call opaque pybind `_CUDA.*` ops with no
meta/fake kernel, so `torch.compile` graph-breaks on them. This OPT-IN submodule re-exposes the same interface
wrapped in `torch.library.custom_op` + `register_fake`, so e.g. `from qutlass.ops import matmul_mxf4_bf16_tn`
gives a drop-in, traceable version of `qutlass.matmul_mxf4_bf16_tn`. `import qutlass` does NOT load this module
and qutlass's own top-level functions are unchanged -- you opt in via `qutlass.ops`.

Each wrapper DELEGATES to the matching `qutlass.*` function (so eager numerics are identical), and adds a
`register_fake` whose shapes are taken from qutlass's own allocation logic (`get_padded_shape_mx/nv` and the
per-op `torch.empty(...)` shapes) so the fakes can't drift from the real kernels. Two notes vs a naive wrapper:
  * the matmul wrappers are pure pass-throughs (scales passed as-is, exactly like `qutlass.matmul_*`);
  * variants that change the output arity (`return_mask`) are separate custom_ops behind a thin Python
    dispatcher, since a custom_op has a fixed number of outputs.

Coverage: the forward-quant (mx/nv/wush) ops + the fp4 matmuls are the inference path (exercised by the repo's
QuantLinear cuda backend -- bit-exact, 0 graph breaks). The mxfp8 matmuls and the backward/transpose ops are
training-only; they are registered with shapes read from the qutlass source but are NOT exercised here, so their
runtime numerics are unvalidated in this repo.
"""

import torch
import qutlass
from qutlass.utils import get_padded_shape_mx, get_padded_shape_nv

_E8M0 = torch.float8_e8m0fnu
_E4M3 = torch.float8_e4m3fn
_FP4X2 = torch.float4_e2m1fn_x2


def _mx_outs(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # (packed e2m1, padded row-major e8m0 scale) -- matches qutlass.fusedQuantizeMx's allocation
    padded_rows, padded_cols = get_padded_shape_mx(a)
    return (a.new_empty((*a.shape[:-1], a.size(-1) // 2), dtype=torch.uint8),
            a.new_empty((padded_rows, padded_cols), dtype=_E8M0))


def _nv_outs(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    padded_rows, padded_cols = get_padded_shape_nv(a)
    return (a.new_empty((*a.shape[:-1], a.size(-1) // 2), dtype=torch.uint8),
            a.new_empty((padded_rows, padded_cols), dtype=_E4M3))


# ===================================================================================================== #
# Forward quantization  (inference)
# ===================================================================================================== #
@torch.library.custom_op('qutlass_ops::fused_quantize_mx_absmax', mutates_args=())
def _fused_quantize_mx_absmax(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return qutlass.fusedQuantizeMx(a, b, method='abs_max')


@_fused_quantize_mx_absmax.register_fake
def _(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _mx_outs(a)


@torch.library.custom_op('qutlass_ops::fused_quantize_mx_quest', mutates_args=())
def _fused_quantize_mx_quest(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return qutlass.fusedQuantizeMx(a, b, method='quest')


@_fused_quantize_mx_quest.register_fake
def _(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _mx_outs(a)


@torch.library.custom_op('qutlass_ops::fused_quantize_mx_quest_mask', mutates_args=())
def _fused_quantize_mx_quest_mask(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return qutlass.fusedQuantizeMx(a, b, method='quest', return_mask=True)


@_fused_quantize_mx_quest_mask.register_fake
def _(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    e2m1, scale = _mx_outs(a)
    return e2m1, scale, a.new_empty((*a.shape[:-1], a.size(-1) // 8), dtype=torch.uint8)


def fusedQuantizeMx(a, b, *, method='quest', return_mask=False):
    """Traceable `qutlass.fusedQuantizeMx`. method in {'quest','abs_max'}; return_mask only with 'quest'."""
    if method == 'abs_max':
        if return_mask:
            raise ValueError("return_mask is only supported for method 'quest'")
        return _fused_quantize_mx_absmax(a, b)
    if method == 'quest':
        return _fused_quantize_mx_quest_mask(a, b) if return_mask else _fused_quantize_mx_quest(a, b)
    raise ValueError(f"invalid method {method!r}, must be 'quest' or 'abs_max'")


@torch.library.custom_op('qutlass_ops::fused_quantize_wush_mx', mutates_args=())
def fusedQuantizeWushMx(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return qutlass.fusedQuantizeWushMx(a, b)


@fusedQuantizeWushMx.register_fake
def _(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _mx_outs(a)


@torch.library.custom_op('qutlass_ops::fused_quantize_nv_absmax', mutates_args=())
def _fused_quantize_nv_absmax(a: torch.Tensor, b: torch.Tensor, global_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return qutlass.fusedQuantizeNv(a, b, global_scale, method='abs_max')


@_fused_quantize_nv_absmax.register_fake
def _(a: torch.Tensor, b: torch.Tensor, global_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _nv_outs(a)


@torch.library.custom_op('qutlass_ops::fused_quantize_nv_quest', mutates_args=())
def _fused_quantize_nv_quest(a: torch.Tensor, b: torch.Tensor, global_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return qutlass.fusedQuantizeNv(a, b, global_scale, method='quest')


@_fused_quantize_nv_quest.register_fake
def _(a: torch.Tensor, b: torch.Tensor, global_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _nv_outs(a)


def fusedQuantizeNv(a, b, global_scale, *, method='abs_max'):
    """Traceable `qutlass.fusedQuantizeNv`. method in {'quest','abs_max'}."""
    if method == 'abs_max':
        return _fused_quantize_nv_absmax(a, b, global_scale)
    if method == 'quest':
        return _fused_quantize_nv_quest(a, b, global_scale)
    raise ValueError(f"invalid method {method!r}, must be 'quest' or 'abs_max'")


# ===================================================================================================== #
# GEMMs  (pass-through; (M, N) bf16 out). cutlass backend.
# ===================================================================================================== #
@torch.library.custom_op('qutlass_ops::matmul_mxf4_bf16_tn', mutates_args=())
def matmul_mxf4_bf16_tn(a: torch.Tensor, b: torch.Tensor, a_sf: torch.Tensor, b_sf: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return qutlass.matmul_mxf4_bf16_tn(a, b, a_sf, b_sf, alpha)


@matmul_mxf4_bf16_tn.register_fake
def _(a: torch.Tensor, b: torch.Tensor, a_sf: torch.Tensor, b_sf: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return a.new_empty((a.size(0), b.size(0)), dtype=torch.bfloat16)


@torch.library.custom_op('qutlass_ops::matmul_ada_mxf4_bf16_tn', mutates_args=())
def matmul_ada_mxf4_bf16_tn(a: torch.Tensor, b: torch.Tensor, a_sf: torch.Tensor, b_sf: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return qutlass.matmul_ada_mxf4_bf16_tn(a, b, a_sf, b_sf, alpha)


@matmul_ada_mxf4_bf16_tn.register_fake
def _(a: torch.Tensor, b: torch.Tensor, a_sf: torch.Tensor, b_sf: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return a.new_empty((a.size(0), b.size(0)), dtype=torch.bfloat16)


@torch.library.custom_op('qutlass_ops::matmul_nvf4_bf16_tn', mutates_args=())
def matmul_nvf4_bf16_tn(a: torch.Tensor, b: torch.Tensor, a_sf: torch.Tensor, b_sf: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return qutlass.matmul_nvf4_bf16_tn(a, b, a_sf, b_sf, alpha)


@matmul_nvf4_bf16_tn.register_fake
def _(a: torch.Tensor, b: torch.Tensor, a_sf: torch.Tensor, b_sf: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return a.new_empty((a.size(0), b.size(0)), dtype=torch.bfloat16)


@torch.library.custom_op('qutlass_ops::matmul_mxf8_bf16_tn', mutates_args=())
def matmul_mxf8_bf16_tn(a: torch.Tensor, b: torch.Tensor, block_scale_a: torch.Tensor, block_scale_b: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return qutlass.matmul_mxf8_bf16_tn(a, b, block_scale_a, block_scale_b, alpha)


@matmul_mxf8_bf16_tn.register_fake
def _(a: torch.Tensor, b: torch.Tensor, block_scale_a: torch.Tensor, block_scale_b: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return a.new_empty((a.size(0), b.size(0)), dtype=torch.bfloat16)


@torch.library.custom_op('qutlass_ops::matmul_mxf8_bf16_nn', mutates_args=())
def matmul_mxf8_bf16_nn(a: torch.Tensor, b: torch.Tensor, block_scale_a: torch.Tensor, block_scale_b: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return qutlass.matmul_mxf8_bf16_nn(a, b, block_scale_a, block_scale_b, alpha)


@matmul_mxf8_bf16_nn.register_fake
def _(a: torch.Tensor, b: torch.Tensor, block_scale_a: torch.Tensor, block_scale_b: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return a.new_empty((a.size(1), b.size(0)), dtype=torch.bfloat16)  # NN operand layout


# ===================================================================================================== #
# Backward / transpose  (training-only; shapes from the qutlass source, unverified in this PTQ repo)
# ===================================================================================================== #
@torch.library.custom_op('qutlass_ops::backward_t_bf16', mutates_args=())
def backward_t_bf16(x: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return qutlass.backward_t_bf16(x, h)


@backward_t_bf16.register_fake
def _(x: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return (x.new_empty((*x.shape[:-2], x.size(-1), x.size(-2) // 2), dtype=_FP4X2),
            x.new_empty((*x.shape[:-2], x.size(-1), x.size(-2) // 32), dtype=_E8M0))


@torch.library.custom_op('qutlass_ops::backward_qt_bf16', mutates_args=())
def backward_qt_bf16(x_e2m1: torch.Tensor, x_e8m0: torch.Tensor, h: torch.Tensor, alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return qutlass.backward_qt_bf16(x_e2m1, x_e8m0, h, alpha)


@backward_qt_bf16.register_fake
def _(x_e2m1: torch.Tensor, x_e8m0: torch.Tensor, h: torch.Tensor, alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return (x_e2m1.new_empty((*x_e2m1.shape[:-2], x_e2m1.size(-1) * 2, x_e2m1.size(-2) // 2), dtype=_FP4X2),
            x_e8m0.new_empty((*x_e8m0.shape[:-2], x_e8m0.size(-1) * 32, x_e8m0.size(-2) // 32), dtype=_E8M0))


@torch.library.custom_op('qutlass_ops::backward_bf16_square_double_mxfp8', mutates_args=())
def backward_bf16_square_double_mxfp8(x_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return qutlass.backward_bf16_square_double_mxfp8(x_bf16)


@backward_bf16_square_double_mxfp8.register_fake
def _(x_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n = x_bf16.size(0), x_bf16.size(1)
    m_pad = m if m % 128 == 0 else ((m - 1) // 128) * 128 + 128  # qutlass pads rows to a multiple of 128
    return (x_bf16.new_empty((m_pad, n), dtype=_E4M3),
            x_bf16.new_empty((m_pad, n // 32), dtype=_E8M0),
            x_bf16.new_empty((n, m_pad // 32), dtype=_E8M0))


@torch.library.custom_op('qutlass_ops::mxfp4_transpose_mxfp8', mutates_args=('scales',))
def mxfp4_transpose_mxfp8(x_fp4: torch.Tensor, scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return qutlass.mxfp4_transpose_mxfp8(x_fp4, scales)  # qutlass pads x_fp4 to 256 rows and writes `scales`


@mxfp4_transpose_mxfp8.register_fake
def _(x_fp4: torch.Tensor, scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m = x_fp4.size(0)
    m_pad = m if m % 256 == 0 else ((m - 1) // 256) * 256 + 256  # qutlass pads rows to a multiple of 256
    cols2 = x_fp4.size(1) * 2
    return (x_fp4.new_empty((cols2, m_pad), dtype=_E4M3),
            x_fp4.new_empty((cols2, m_pad // 32), dtype=_E8M0))
