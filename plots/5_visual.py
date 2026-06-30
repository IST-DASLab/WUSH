import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import scipy
import torch

from plot_utils import CmapLegend, HandlerCmap

get_normalized_hadamard_transform = lambda size, dtype=torch.float64, device=torch.device('cpu'): torch.as_tensor(scipy.linalg.hadamard(size), dtype=dtype, device=device) * size ** -.5

torch.set_float32_matmul_precision('highest')  # exact fp32 matmuls for reproducible transforms

plt.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'path',
    'svg.hashsalt': 'fixed-salt',
})
strip_svg_meta: dict[str, None] = {k: None for k in ('Creator', 'Date', 'Format', 'Type')}
strip_pdf_meta: dict[str, None] = {k: None for k in ('Title', 'Author', 'Subject', 'Keywords', 'Creator', 'Producer', 'CreationDate', 'ModDate', 'Trapped')}


def get_transform(
        basis: torch.Tensor,
        y: torch.Tensor,
        transform_type: str = 'identity',
) -> torch.Tensor:
    """
    basis: (..., B, B)
    y: (..., B, R)
    """
    dtype, device = basis.dtype, basis.device
    transform_block_size: int = basis.size(-1)

    match transform_type:
        case 'identity':
            transform: torch.Tensor = torch.eye(transform_block_size, dtype=dtype, device=device)  # (B, B)
        case 'random_rotation':
            transform: torch.Tensor = torch.linalg.qr(torch.randn(*basis.shape[:-2], transform_block_size, transform_block_size, dtype=dtype, device=device), mode='reduced').Q  # (..., B, B)
        case 'hadamard':
            transform: torch.Tensor = get_normalized_hadamard_transform(transform_block_size, dtype=dtype, device=device)  # (B, B)
        case 'hsuw' | 'suw':
            if transform_type == 'suw':
                hadamard: torch.Tensor = torch.eye(transform_block_size, dtype=dtype, device=device)  # (B, B)
            else:
                hadamard: torch.Tensor = get_normalized_hadamard_transform(transform_block_size, dtype=dtype, device=device)  # (B, B)
            v, s, uh = torch.linalg.svd(y, full_matrices=False)  # (..., B, B), (..., B), (..., B, R)
            s *= y.size(-1) ** -.5  # (..., B)
            transform: torch.Tensor = torch.linalg.solve_triangular(
                basis.transpose(-2, -1),  # (..., B, B)
                hadamard * s[..., None, :] ** .5 @ v.transpose(-2, -1),  # (..., B, B)
                upper=False,
                left=False,
                unitriangular=False,
            )  # (..., B, B), T_{hsvx}^{-\top}, transform = hadamard @ diag(s ** .5) @ v.t() @ basis.t().inv()
        case _:
            raise NotImplementedError

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


def plot_visual():
    torch.manual_seed(0)

    lim_in, lim_channel = 512, 256
    # 5_visual.pt is a pre-extracted slice [weight[:256, :512], activation[:1, :256, :512]] (bf16) of the full weights/activations tensor.
    weight, activation = torch.load('5_visual.pt', weights_only=True)
    weight, activation = weight[:lim_channel, :lim_in].to(dtype=torch.float64), activation[0, :lim_channel, :lim_in].to(dtype=torch.float64)

    fig = plt.figure(figsize=(24, 8))

    smooth_quant_alpha: float = .5
    transform_smooth_quant: torch.Tensor = torch.linalg.vector_norm(activation, ord=torch.inf, dim=-2).pow(smooth_quant_alpha) / torch.linalg.vector_norm(weight, ord=torch.inf, dim=-2).pow(1. - smooth_quant_alpha)
    weight_smooth_quant_transformed: torch.Tensor = weight * transform_smooth_quant
    activation_smooth_quant_transformed: torch.Tensor = activation / transform_smooth_quant

    block_size: int = 32
    basis: torch.Tensor = torch.linalg.cholesky_ex(activation.unflatten(dim=-1, sizes=(-1, block_size)).permute(1, 2, 0) @ activation.unflatten(dim=-1, sizes=(-1, block_size)).permute(1, 0, 2), upper=True, check_errors=False).L * activation.size(-2) ** -.5
    y: torch.Tensor = basis @ weight.unflatten(dim=-1, sizes=(-1, block_size)).permute(1, 2, 0)

    for i, transform_type in enumerate(['identity', 'random_rotation', 'hadamard', 'suw', 'hsuw', 'smooth_quant']):
        if transform_type == 'smooth_quant':
            weight_transformed = weight_smooth_quant_transformed
            activation_transformed = activation_smooth_quant_transformed
        else:
            transform_a: torch.Tensor = get_transform(basis, y, transform_type=transform_type)
            transform_w: torch.Tensor = torch.linalg.inv(transform_a.transpose(-2, -1))
            weight_transformed: torch.Tensor = block_transform(weight, transform=transform_w)
            activation_transformed: torch.Tensor = block_transform(activation, transform=transform_a)
        weight_transformed_rms: torch.Tensor = torch.linalg.vector_norm(weight_transformed, dim=-2) * weight_transformed.size(-2) ** -.5
        activation_transformed_rms: torch.Tensor = torch.linalg.vector_norm(activation_transformed, dim=-2) * activation_transformed.size(-2) ** -.5

        for j, (Z, Z_rms) in enumerate(zip([weight_transformed.abs().numpy(), activation_transformed.abs().numpy()], [weight_transformed_rms.numpy(), activation_transformed_rms.numpy()])):
            stride: int = 1

            Z = Z[:lim_in, :]
            ax = fig.add_subplot(2, 6, j * 6 + i + 1, projection='3d')
            surface = ax.plot_surface(
                *np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]), indexing='ij'), Z.T,
                cmap='viridis',
                rstride=1, cstride=stride,
                linewidth=0.,
                antialiased=True,
                shade=False,
            )
            surface.set_rasterized(True)
            ax.plot(np.arange(Z.shape[1]), np.full(Z.shape[1], -1), Z_rms, color='red', linewidth=1., zorder=2.5)
            ax.tick_params(axis='x', labelsize=12, pad=2)
            ax.tick_params(axis='y', labelsize=12, pad=2)
            ax.tick_params(axis='z', labelsize=12, pad=2)
            ax.set_xlabel('Input Channel', fontsize=12, labelpad=4)
            ax.set_ylabel(f"{['Output Channel', 'Token'][j]}", fontsize=12, labelpad=4)
            ax.set_zlabel('Absolute Value', fontsize=12, labelpad=4)
            ax.set_title(f"{['Weight', 'Activation'][j]} - {['I', 'R', 'H', 'WUS', 'WUSH', 'SmoothQuant'][i]}", fontsize=12, pad=0)

            for a in (ax.xaxis, ax.yaxis, ax.zaxis):
                a.pane.set_facecolor((1., 1., 1., 1.))
                a.pane.set_alpha(1.)
            ax.set_facecolor((1., 1., 1., 0.))

    handles = {}
    handles["Magnitude"] = CmapLegend(mpl.cm.viridis, mpl.colors.TwoSlopeNorm(vmin=-2., vcenter=0., vmax=2.), n=80, orientation='horizontal')
    handles["RMS (Root Mean Square)"], = ax.plot(100., 100., color='red', linewidth=1., zorder=2.5)
    fig.legend(handles.values(), handles.keys(), ncol=len(handles), loc='upper center', bbox_to_anchor=(.5, .0), fontsize=12, frameon=True, framealpha=1., handler_map={CmapLegend: HandlerCmap()})

    fig.set_facecolor((1., 1., 1., 0.))
    # fig.tight_layout()  # disabled to prevent title overlapping with plot
    fig.savefig('5_visual.svg', bbox_inches='tight', pad_inches=.5, transparent=False, metadata=strip_svg_meta)
    fig.savefig('5_visual.pdf', bbox_inches='tight', pad_inches=.5, transparent=False, metadata=strip_pdf_meta)
    # fig.show()


if __name__ == '__main__':
    plot_visual()
