import itertools

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Ellipse, Polygon
import torch

from plot_utils import get_slopes_from_transform, CmapLegend, HandlerCmap

plt.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'path',
    'svg.hashsalt': 'fixed-salt',
})
strip_svg_meta: dict[str, None] = {k: None for k in ('Creator', 'Date', 'Format', 'Type')}
strip_pdf_meta: dict[str, None] = {k: None for k in ('Title', 'Author', 'Subject', 'Keywords', 'Creator', 'Producer', 'CreationDate', 'ModDate', 'Trapped')}

font = mpl.font_manager.FontProperties(size=10)
font_legend = mpl.font_manager.FontProperties(size=10)

new_bwr = mpl.colors.LinearSegmentedColormap.from_list(
    "new_bwr",
    ["dodgerblue", "white", "red"],   # or ["#1E90FF", "white", "red"]
    N=256
)


def get_rot_matrix_2d(theta: torch.Tensor) -> torch.Tensor:
    """

    :param theta: (...)
    :return: (..., D=2, D=2)
    """
    return torch.stack([
        torch.stack([theta.cos(), -theta.sin()], dim=-1),
        torch.stack([theta.sin(), theta.cos()], dim=-1),
    ], dim=-2)


def get_error_corners(
        points_xy: torch.Tensor,
        quant_format: str,
        transform: torch.Tensor = None,
) -> torch.Tensor:
    """

    :param points_xy: (..., D=2)
    :param quant_format: str
    :param transform: (..., D=2, D=2)
    :return: (..., 2^(D=2), D=2)
    """
    corner_coords_xy = torch.as_tensor([[1., 1.], [-1., 1.], [-1., -1.], [1., -1.]], dtype=points_xy.dtype)  # (2^D, D)

    if transform is not None:
        points = points_xy @ transform.transpose(-2, -1)
    else:
        points = points_xy.clone()

    match quant_format:
        case 'fp':
            error_corner_coords_xy = corner_coords_xy * points.abs()[..., None, :]
        case 'int':
            error_corner_coords_xy = corner_coords_xy * points.abs().amax(dim=-1)[..., None, None]
        case _:
            raise NotImplementedError

    if transform is not None:
        error_corner_coords_xy = torch.linalg.solve_ex(transform.transpose(-2, -1), error_corner_coords_xy, left=False, check_errors=False).result
    else:
        error_corner_coords_xy = error_corner_coords_xy.clone()
    return error_corner_coords_xy


def get_error_norm(error_corner_coords_xy: torch.Tensor) -> torch.Tensor:
    """

    :param error_corner_coords_xy: (..., D=2)
    :return: (...)
    """
    circle_radii = ((
                     (error_corner_coords_xy[..., 1, :] - error_corner_coords_xy[..., 0, :]).pow(2.).sum(dim=-1)
                   + (error_corner_coords_xy[..., 3, :] - error_corner_coords_xy[..., 0, :]).pow(2.).sum(dim=-1)
             ) / 12.) ** .5
    return circle_radii


def draw_basis_grid(
        ax: plt.Axes,
        basis_xy: torch.Tensor,
        transform: torch.Tensor = None,
        frame_scale: float = 1.
) -> None:
    """

    :param ax:
    :param basis_xy: (D=2, D=2)
    :param transform: (D=2, D=2)
    :param frame_scale: float
    :return: None
    """
    dtype = basis_xy.dtype
    if transform is not None:
        basis = basis_xy @ transform.transpose(-2, -1)
    else:
        basis = basis_xy.clone()
    corner_coords_xy = torch.as_tensor([[1., 1.], [-1., 1.], [-1., -1.], [1., -1.]], dtype=dtype)  # (2^D, D)
    lattice_sizes = torch.linalg.solve(basis, corner_coords_xy * frame_scale, left=False).abs().amax(dim=0).ceil().to(dtype=torch.int64) + 1  # (D), + 1 to prevent missing voronoi edges
    lattice_ab = torch.stack(torch.meshgrid([torch.arange(-s, s + 1, dtype=dtype) for s in lattice_sizes], indexing='ij'), dim=-1)  # (..., D)
    lattice_xy = lattice_ab @ basis  # (..., D)
    ax.plot(lattice_xy[[0, -1], :, 0], lattice_xy[[0, -1], :, 1], color='black', linestyle=':', linewidth=1., zorder=2.)
    ax.plot(lattice_xy[:, [0, -1], 0].transpose(0, 1), lattice_xy[:, [0, -1], 1].transpose(0, 1), color='black', linestyle=':', linewidth=1., zorder=2.)


def draw_covariance_ellipse(
        ax: plt.Axes,
        covariance: torch.Tensor,
        transform: torch.Tensor = None,
        ellipse_scale: float = 1.,
) -> None:
    """

    :param ax:
    :param covariance: (D=2, D=2)
    :param transform: (D=2, D=2)
    :param ellipse_scale: float
    :return:
    """
    if transform is not None:
        cov = transform @ covariance @ transform.transpose(-2, -1)
    else:
        cov = covariance.clone()
    eigenvalues, eigenvectors = torch.linalg.eigh(cov, UPLO='L')
    # eigenvectors @ eigenvalues.diag() @ eigenvectors.t() = covariance, ascending eigenvalues
    # eigenvalues, eigenvectors = eigenvalues.flip(dims=(-1,)), eigenvectors.flip(dims=(-1,))
    ellipse_a, ellipse_b = eigenvalues ** .5
    ellipse_theta = eigenvectors[1, 0].atan2(eigenvectors[0, 0])
    # if torch.linalg.det(eigenvectors) < 0.:
    #     reflection = make_rot_matrix_2d(-ellipse_theta) @ eigenvectors
    #     ellipse_theta = reflection[0, 1].atan2(eigenvectors[0, 0]) - ellipse_theta
    ellipse = Ellipse(
        xy=(0., 0.),
        width=2. * ellipse_scale * ellipse_a,
        height=2. * ellipse_scale * ellipse_b,
        angle=ellipse_theta.rad2deg(),
        alpha=1., edgecolor='black', facecolor=None, fill=False, linestyle='-', linewidth=1., zorder=2.3,
    )
    ax.add_patch(ellipse)


def draw_points(
        ax: plt.Axes,
        points_xy: torch.Tensor,
        transform: torch.Tensor = None,
        colors: tuple | list = None,
) -> None:
    """

    :param ax:
    :param points_xy: (..., D=2)
    :param transform: (..., D=2, D=2)
    :param colors: list
    :return: None
    """
    if transform is not None:
        points = points_xy @ transform.transpose(-2, -1)
    else:
        points = points_xy.clone()

    points = points.reshape(-1, points.size(-1))
    ax.scatter(
        points[:, 0],
        points[:, 1],
        color=colors[:len(points)] if colors is not None else None,
        alpha=1., marker='o', s=36., zorder=2.5,
    )


def draw_error_zones(
        ax: plt.Axes,
        points_xy: torch.Tensor,
        error_corner_coords_xy: torch.Tensor,
        transform: torch.Tensor = None,
        error_scale: float = 1.,
        colors: tuple | list = None,
) -> None:
    """

    :param ax:
    :param points_xy: (..., D=2)
    :param error_corner_coords_xy: (..., 2^(D=2), D=2)
    :param transform: (..., D=2, D=2)
    :param error_scale: float
    :param colors: list
    :return: None
    """
    if transform is not None:
        points = points_xy @ transform.transpose(-2, -1)
        error_corner_coords = error_corner_coords_xy @ transform.transpose(-2, -1)
    else:
        points = points_xy.clone()
        error_corner_coords = error_corner_coords_xy.clone()

    parallelogram_coords = points[..., None, :] + error_corner_coords * error_scale
    circle_radii = get_error_norm(error_corner_coords) * error_scale

    points = points.reshape(-1, points.size(-1))
    circle_radii = circle_radii.flatten()
    parallelogram_coords = parallelogram_coords.reshape(-1, *parallelogram_coords.shape[-2:])

    for i in range(len(points)):
        parallelogram = Polygon(
            xy=parallelogram_coords[i],
            facecolor=colors[i] if colors is not None else None,
            alpha=.25, edgecolor=None, fill=True, linestyle='-', linewidth=0., zorder=2.4,
        )
        circle = Circle(
            xy=points[i],
            radius=circle_radii[i],
            edgecolor=colors[i] if colors is not None else None,
            alpha=1., facecolor=None, fill=False, linestyle='-', linewidth=1., zorder=2.5,
        )
        ax.add_patch(parallelogram)
        ax.add_patch(circle)


def draw_diff(
        ax: plt.Axes,
        quant_format: str,
        transform: torch.Tensor = None,
        is_transformed_space: bool = False,
        frame_scale: float = 1.
) -> None:
    """

    :param ax:
    :param quant_format: str
    :param transform: (D=2, D=2)
    :param frame_scale: float
    :param is_transformed_space: bool
    :return: None
    """
    dtype = transform.dtype
    lattice_xy = torch.stack(torch.meshgrid([torch.linspace(-frame_scale, frame_scale, 100, dtype=dtype)] * 2, indexing='ij'), dim=-1).flatten(end_dim=-2)  # (..., D)
    if is_transformed_space:
        lattice_xy_original_space = torch.linalg.solve_ex(transform.transpose(-2, -1), lattice_xy, left=False, check_errors=False).result
    else:
        lattice_xy_original_space = lattice_xy.clone()
    error_corner_coords_xy_original = get_error_corners(
        points_xy=lattice_xy_original_space,
        quant_format=quant_format,
        transform=None,
    )
    error_corner_coords_xy_transformed = get_error_corners(
        points_xy=lattice_xy_original_space,
        quant_format=quant_format,
        transform=transform,
    )
    circle_radii_original = get_error_norm(error_corner_coords_xy_original)
    circle_radii_transformed = get_error_norm(error_corner_coords_xy_transformed)
    c = circle_radii_transformed - circle_radii_original

    tri = mpl.tri.Triangulation(lattice_xy[..., 0], lattice_xy[..., 1])
    norm = mpl.colors.TwoSlopeNorm(vmin=-2., vcenter=0., vmax=2.)
    heatmap = ax.tripcolor(tri, c.flatten(), norm=norm, cmap='bwr', shading='gouraud', zorder=1.)
    heatmap.set_rasterized(True)


def draw_separation_lines(
        ax: plt.Axes,
        quant_format: str,
        transform: torch.Tensor,
        is_transformed_space: bool = False,
) -> None:
    """

    :param ax:
    :param quant_format: str
    :param transform: (D=2, D=2)
    :param ellipse_scale: float
    :param is_transformed_space: bool
    :return: None
    """
    match quant_format:
        case 'fp':
            if is_transformed_space:
                transform = torch.eye(2, dtype=torch.float64)
            scale = torch.as_tensor([-1000., 1000.], dtype=transform.dtype)
            ax.plot(-transform[0, 1] * scale, transform[0, 0] * scale, color='black', linestyle='--', linewidth=1., zorder=2.)
            ax.plot(-transform[1, 1] * scale, transform[1, 0] * scale, color='black', linestyle='--', linewidth=1., zorder=2.)
        case 'int':
            # return
            slopes = get_slopes_from_transform(transform)
            s = 1000.
            end_points = torch.as_tensor([(s, s * slope) if torch.as_tensor(slope).isfinite() else (0., s) for slope in slopes])
            if is_transformed_space:
                end_points = end_points @ transform.transpose(-2, -1)
            for end_point in end_points:
                ax.plot([-end_point[0], end_point[0]], [-end_point[1], end_point[1]], color='black', linestyle='--', linewidth=1., zorder=2.)
        case _:
            raise NotImplementedError


def plot_2d() -> None:
    dtype = torch.float64

    points_xy = torch.as_tensor([[.92, .22], [.38, -.55]], dtype=dtype)  # (N, D), row vectors
    # points_xy = torch.randn(10, 2, dtype=dtype)  # (N, D), row vectors

    ellipse_a = torch.as_tensor(1e0, dtype=dtype)
    ellipse_b = torch.as_tensor(3e-1, dtype=dtype)
    ellipse_theta = torch.as_tensor(15., dtype=dtype).deg2rad()

    frame_scale = 1.5
    grid_scale = .5
    ellipse_scale = 1.15
    error_scale = .5

    basis_xy = torch.eye(2, dtype=dtype) * grid_scale  # (N, D), row vectors
    singular = torch.stack([ellipse_a, ellipse_b], dim=-1)
    ellipse_rotation = get_rot_matrix_2d(theta=ellipse_theta)
    covariance = ellipse_rotation * singular ** 2. @ ellipse_rotation.transpose(-2, -1)

    rand_rot = torch.linalg.qr(torch.randn(2, 2, dtype=dtype), mode='reduced').Q
    # hadamard = torch.as_tensor(scipy.linalg.hadamard(2), dtype=dtype) * 2 ** -.5
    # hadamard[1, :] *= -1.
    hadamard = get_rot_matrix_2d(torch.as_tensor(45., dtype=dtype).deg2rad())
    u = ellipse_rotation.transpose(-2, -1)
    hu = hadamard @ ellipse_rotation.transpose(-2, -1)
    ssu = ellipse_rotation.transpose(-2, -1) / singular[..., None]
    su = ellipse_rotation.transpose(-2, -1) * singular[..., None] ** -.5
    hsu = hadamard * singular ** -.5 @ ellipse_rotation.transpose(-2, -1)

    # colors = '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#000000'
    colors = 'green', 'limegreen'

    fig, axs = plt.subplots(nrows=6, ncols=4, figsize=(8., 11.8))

    for i, ax in enumerate(axs.flat):
        ax.axvline(x=0., color='black', linestyle='-', linewidth=1., zorder=2.)
        ax.axhline(y=0., color='black', linestyle='-', linewidth=1., zorder=2.)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-frame_scale, frame_scale)
        ax.set_ylim(-frame_scale, frame_scale)
        ax.set_aspect('equal')
        ax.set_facecolor((1., 1., 1., 1.))

    for i, transform in enumerate([None, hadamard, hu, ssu, su, hsu]):
        for j, (quant_format, trans) in enumerate(itertools.product(['fp', 'int'], [None, transform])):
            draw_basis_grid(
                ax=axs[i, j],
                basis_xy=basis_xy,
                transform=trans,
                frame_scale=frame_scale,
            )
            draw_covariance_ellipse(
                ax=axs[i, j],
                covariance=covariance,
                transform=trans,
                ellipse_scale=ellipse_scale,
            )
            draw_points(
                ax=axs[i, j],
                points_xy=points_xy,
                transform=trans,
                colors=colors,
            )
            error_corner_coords_xy = get_error_corners(
                points_xy=points_xy,
                quant_format=quant_format,
                transform=transform,
            )
            draw_error_zones(
                ax=axs[i, j],
                points_xy=points_xy,
                error_corner_coords_xy=error_corner_coords_xy,
                transform=trans,
                error_scale=error_scale,
                colors=colors,
            )
            if (quant_format == 'int' and transform is not None) or (quant_format == 'fp' and i == 5):
                draw_diff(
                    ax=axs[i, j],
                    quant_format=quant_format,
                    transform=transform,
                    is_transformed_space=trans is not None,
                    frame_scale=frame_scale,
                )
                draw_separation_lines(
                    ax=axs[i, j],
                    quant_format=quant_format,
                    transform=transform,
                    is_transformed_space=trans is not None,
                )

    axs[0, 0].set_ylabel(r'Identity $\boldsymbol{T} = \mathbf{I}$', fontproperties=font)
    axs[1, 0].set_ylabel(r'Hadamard $\boldsymbol{T} = \boldsymbol{H}$', fontproperties=font)
    axs[2, 0].set_ylabel(r'Calibrated Hadamard $\boldsymbol{T} = \boldsymbol{H} \boldsymbol{U}^\top$', fontproperties=font)
    axs[3, 0].set_ylabel(r'Whitening $\boldsymbol{T} = \boldsymbol{S}^{-1} \boldsymbol{U}^\top$', fontproperties=font)
    axs[4, 0].set_ylabel(r'WUS $\boldsymbol{T} = \boldsymbol{S}^{-1/2} \boldsymbol{U}^\top$', fontproperties=font)
    axs[5, 0].set_ylabel(r'WUSH $\boldsymbol{T} = \boldsymbol{H} \boldsymbol{S}^{-1/2} \boldsymbol{U}^\top$', fontproperties=font)
    axs[0, 0].set_title('FP (Original Space)', fontproperties=font)
    axs[0, 1].set_title('FP (Transformed Space)', fontproperties=font)
    axs[0, 2].set_title('INT (Original Space)', fontproperties=font)
    axs[0, 3].set_title('INT (Transformed Space)', fontproperties=font)

    ax = axs[0, 0]
    handles = {}
    handles["Coordinate Axis"], = ax.plot(100., 100., color='black', linestyle='-', linewidth=1.)
    handles["Example Point 1"] = ax.scatter(100., 100., alpha=1., color=colors[0], marker='o', s=36.,)
    handles["Random Noise Distributing Zone of Point 1"], = ax.fill(100., 100., alpha=.25, facecolor=colors[0], edgecolor=None, linestyle='-', linewidth=0.)
    handles["Expected Error (Shown as Radius) of Point 1"] = ax.scatter(100., 100., alpha=1., facecolors='none', edgecolors=colors[0], s=81., marker='o', linewidths=1.)
    handles["Probabilistic Contour of Data Distribution"] = ax.scatter(100., 100., alpha=1., facecolors='none', edgecolors='black', s=81., marker='o', linewidths=1.)
    handles["Equilibrium Line of Expected Error Change"], = ax.plot(100., 100., color='black', linestyle='--', linewidth=1.)
    handles["Basis Grid"], = ax.plot(100., 100., color='black', linestyle=':', linewidth=1.)
    handles["Example Point 2"] = ax.scatter(100., 100., alpha=1., color=colors[1], marker='o', s=36.,)
    handles["Random Noise Distributing Zone of Point 2"], = ax.fill(100., 100., alpha=.25, facecolor=colors[1], edgecolor=None, linestyle='-', linewidth=0.)
    handles["Expected Error (Shown as Radius) of Point 2"] = ax.scatter(100., 100., alpha=1., facecolors='none', edgecolors=colors[1], s=81., marker='o', linewidths=1.)
    handles["Change of Expected Error after Transform\n(Blue ↓  White -  Red ↑)"] = CmapLegend(mpl.cm.bwr, mpl.colors.TwoSlopeNorm(vmin=-2., vcenter=0., vmax=2.), n=80, orientation='horizontal')

    fig.legend(handles.values(), handles.keys(), loc='upper center', ncol=2, framealpha=1., bbox_to_anchor=(.5, .005), prop=font_legend, handler_map={CmapLegend: HandlerCmap()})
    fig.set_facecolor((1., 1., 1., 0.))
    fig.tight_layout()
    fig.savefig('1_toy_example.svg', bbox_inches='tight', pad_inches=.01, transparent=False, metadata=strip_svg_meta)
    fig.savefig('1_toy_example.pdf', bbox_inches='tight', pad_inches=.01, transparent=False, metadata=strip_pdf_meta)
    fig.show()
    # fig.clf()


if __name__ == '__main__':
    plot_2d()
