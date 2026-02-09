import torch
import re
from inference_lib.src.fp_quant.module.triton.mxfp4 import mxfp4_forward_kernel_wrapper
from inference_lib.src.fp_quant.module.triton.nvfp4 import nvfp4_forward_kernel_wrapper

def pseudoquantize_fp(
    x: torch.Tensor,
    dtype: str,
) -> torch.Tensor:

    class FPQuantDtype:
        MXFP4 = "mxfp4"
        NVFP4 = "nvfp4"
        MXFP8 = "mxfp8"
        BF16 = "bf16"

    forward_method = "abs_max"

    if dtype == FPQuantDtype.MXFP4:
        hadamard_matrix = torch.eye(32, dtype=x.dtype, device=x.device)

        if forward_method == "quest":
            gaussian_scale = 2.92247856 / 6.0
            quest = True
        elif forward_method == "abs_max":
            gaussian_scale = 3.0 / 4.0
            quest = False
        else:
            raise ValueError(f"Unsupported forward method: {forward_method}")

        x_dequantized, mask = mxfp4_forward_kernel_wrapper(
            x,
            hadamard_matrix,
            return_clip_mask=True,
            quest=quest,
            gaussian_scale=gaussian_scale,
        )
        return x_dequantized#, mask
    elif dtype == FPQuantDtype.NVFP4:
        hadamard_matrix = torch.eye(16, dtype=x.dtype, device=x.device)

        assert forward_method == "abs_max", "NVFP4 only supports abs_max method"
        # global_scale = None
        global_scale = torch.as_tensor(10., dtype=x.dtype, device=x.device)
        x_dequantized = nvfp4_forward_kernel_wrapper(
            x,
            hadamard_matrix,
            global_scale,
        )
        return x_dequantized#, torch.ones_like(x_dequantized, dtype=torch.bool)
    elif dtype == FPQuantDtype.MXFP8:
        raise NotImplementedError("MXFP8 is not supported for forward quantization yet")
    else:
        raise ValueError(f"Unsupported forward dtype: {dtype}")


def pseudoquantize_int(
    x: torch.Tensor,
    quant_bitwidth: int,
    quant_group_size: int,
) -> torch.Tensor:
    optimal_gaussian_scales: dict[int | float, float] = {
        1: 0.7978845587140913,
        1.585: 1.2240089519030855,
        2: 1.4935346200015913,
        3: 2.051068354131873,
        4: 2.513930578568423,
        5: 2.9160938834961225,
        6: 3.276597282593217,
        7: 3.6010497188221655,
        8: 3.884938678807525,
    }

    quant_group_size: int = quant_group_size if quant_group_size > 0 else x.size(-1)
    x_grouped: torch.Tensor = x.unflatten(dim=-1, sizes=(-1, quant_group_size))  # (..., C//G, G)
    scale: torch.Tensor = (
        optimal_gaussian_scales[quant_bitwidth] * 2. / (2 ** quant_bitwidth - 1)
        * quant_group_size ** -.5 * torch.linalg.vector_norm(x_grouped, dim=-1, keepdim=True)
    ).clamp(min=torch.finfo(x.dtype).eps)  # (..., C//G, 1)
    x_int: torch.Tensor = (x_grouped / scale).floor().clamp(- 2 ** (quant_bitwidth - 1), 2 ** (quant_bitwidth - 1) - 1)  # (..., C//G, G)
    x_grouped_fake_quant: torch.Tensor = (x_int + .5) * scale  # (..., C//G, G)
    x_fake_quant: torch.Tensor = x_grouped_fake_quant.flatten(start_dim=-2)  # (..., C)
    return x_fake_quant  # (..., C)


def pseudoquantize(
    x: torch.Tensor,
    quant_type: str,
) -> torch.Tensor:
    match = re.match(r"int(\d+)_g(\d+)", quant_type)
    if match:
        quant_bitwidth = int(match.group(1))
        quant_group_size = int(match.group(2))
        x_fake_quant = pseudoquantize_int(x, quant_bitwidth=quant_bitwidth, quant_group_size=quant_group_size)
    else:
        x_fake_quant = pseudoquantize_fp(x, dtype=quant_type)
    return x_fake_quant.to(dtype=x.dtype)


class QuantLinear(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, activation_quantizer: torch.nn.Module = None) -> None:
        super().__init__()
        self.weight: torch.Tensor = torch.nn.Parameter(weight, requires_grad=False)  # (R, C)
        self.activation_quantizer: torch.nn.Module | None = activation_quantizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_quantizer is not None:
            x_fake_quant: torch.Tensor = self.activation_quantizer(x)  # (..., C)
        else:
            x_fake_quant: torch.Tensor = x  # (..., C)
        y: torch.Tensor = x_fake_quant @ self.weight.transpose(-2, -1)  # (..., R)
        return y

class BlockTransformQuantizer(torch.nn.Module):
    def __init__(self, transform: torch.Tensor | None, quant_type: str) -> None:
        super().__init__()
        if transform is not None:
            # (C//G, G, G). 
            self.register_buffer("transform", transform, persistent=True)
        else:
            self.transform = None
        self.quant_type = quant_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., C)  ;  transform: (C//G, G, G)
        """
        compute_dtype = x.dtype  
        y = x.to(dtype=compute_dtype, copy=False)

        if self.transform is not None:
            G = self.transform.size(-1)          
            C = y.size(-1)
            assert C % G == 0, "C must be divisible by group size G"
            N = C // G                          

            # Reshape once: (..., N, G)
            y = y.reshape(*y.shape[:-1], N, G)

            Tt = self.transform.transpose(-2, -1).to(dtype=compute_dtype)
            # einsum keeps memory tight: result (..., N, G)
            y = torch.einsum('...ng,ngh->...nh', y, Tt)

            # Flatten back to (..., C)
            y = y.reshape(*y.shape[:-2], C)

        
            
        y = pseudoquantize(y, quant_type=self.quant_type).to(dtype=x.dtype, copy=False)
        return y
