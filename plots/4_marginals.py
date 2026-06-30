import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm

from plot_utils import CmapLegend, HandlerCmap

plt.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'path',
    'svg.hashsalt': 'fixed-salt',
})
strip_svg_meta: dict[str, None] = {k: None for k in ('Creator', 'Date', 'Format', 'Type')}
strip_pdf_meta: dict[str, None] = {k: None for k in ('Title', 'Author', 'Subject', 'Keywords', 'Creator', 'Producer', 'CreationDate', 'ModDate', 'Trapped')}

font = mpl.font_manager.FontProperties(size=11)
font_legend = mpl.font_manager.FontProperties(size=11)


def _rotmat(deg: float) -> np.ndarray:
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s],
                     [s,  c]], dtype=float)


def _gauss_pdf_1d(t, mu, var):
    t = np.asarray(t, dtype=float)
    mu = float(mu)
    var = float(var)
    if var <= 0:
        raise ValueError("Variance must be > 0.")
    return (1.0 / np.sqrt(2.0 * np.pi * var)) * np.exp(-0.5 * (t - mu) ** 2 / var)


def _gauss_pdf_2d(X, Y, mu, Sigma):
    mu = np.asarray(mu, dtype=float).reshape(2,)
    Sigma = np.asarray(Sigma, dtype=float)
    if Sigma.shape != (2, 2):
        raise ValueError("Sigma must be 2x2.")
    if not np.allclose(Sigma, Sigma.T, atol=1e-12):
        raise ValueError("Sigma must be symmetric.")
    if np.min(np.linalg.eigvalsh(Sigma)) <= 0:
        raise ValueError("Sigma must be positive definite.")

    inv = np.linalg.inv(Sigma)
    det = np.linalg.det(Sigma)

    dx = X - mu[0]
    dy = Y - mu[1]
    q = inv[0, 0] * dx * dx + 2.0 * inv[0, 1] * dx * dy + inv[1, 1] * dy * dy
    norm = 1.0 / (2.0 * np.pi * np.sqrt(det))
    return norm * np.exp(-0.5 * q)


def _plot_joint_3d(
    ax, Sigma, mu, xlim, ylim,
    n=161, elev=23, azim=-58,
    color_px="C0", color_py="C1",
    add_ellipse_contours=True,
    contour_r2=(0.5, 1.0, 2.0, 3.0, 4.5, 6.0),  # Mahalanobis radii^2
    contour_color="0.25",
):
    x = np.linspace(*xlim, n)
    y = np.linspace(*ylim, n)
    X, Y = np.meshgrid(x, y)
    Z = _gauss_pdf_2d(X, Y, mu, Sigma)

    px = _gauss_pdf_1d(x, mu[0], Sigma[0, 0])
    py = _gauss_pdf_1d(y, mu[1], Sigma[1, 1])

    ax.view_init(elev=elev, azim=azim)

    surf = ax.plot_surface(X, Y, Z, cmap=cm.Greens, linewidth=0, antialiased=True, alpha=0.95)
    surf.set_rasterized(True)

    # Ellipse contours (projected onto the floor z=0)
    if add_ellipse_contours:
        det = np.linalg.det(Sigma)
        zmax = 1.0 / (2.0 * np.pi * np.sqrt(det))
        levels = np.array([zmax * np.exp(-0.5 * r2) for r2 in contour_r2])
        # keep only levels within Z range to avoid warnings
        levels = levels[(levels > Z.min()) & (levels < Z.max())]
        if levels.size:
            ax.contour(
                X, Y, Z,
                levels=np.sort(levels),
                zdir="z",
                offset=0.0,
                colors=contour_color,
                linewidths=1.2,
            )

    # Floor grid
    step = max(1, n // 20)
    wire = ax.plot_wireframe(
        X[::step, ::step], Y[::step, ::step], np.zeros_like(Z)[::step, ::step],
        color="0.7", linewidth=0.6, alpha=0.7
    )
    wire.set_rasterized(True)

    # Marginals on walls (colors match 2D subplot)
    ax.plot(x, np.full_like(x, ylim[1]), px, color=color_px, lw=2.)        # p(x)
    ax.plot(np.full_like(y, xlim[0]), y, py, color=color_py, lw=2.)        # p(y)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    # ax.set_zlim(0, max(Z.max(), px.max(), py.max()) * 1.05)
    # ax.set_xlabel(r"$x$", labelpad=8)
    # ax.set_ylabel(r"$y$", labelpad=8)
    # ax.set_zlabel(r"$p$", labelpad=6)

    ax.set_zlim(0., 1.5)
    ax.set_zticks(np.linspace(0., 1.5, 6))
    ax.set_xlabel(r"$x$", labelpad=-15, fontproperties=font)
    ax.set_ylabel(r"$y$", labelpad=-15, fontproperties=font)
    ax.set_zlabel(r"$p$", labelpad=-15, fontproperties=font)

    # Hide tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Remove the little tick stubs in 3D (keep tick locations -> grid stays)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a._axinfo["tick"]["inward_factor"] = 0.0
        a._axinfo["tick"]["outward_factor"] = 0.0
        a.pane.set_facecolor((1., 1., 1., 1.))
        a.pane.set_alpha(1.)
    ax.set_facecolor((1., 1., 1., 0.))

    return x, y, px, py


def _plot_marginals_2d(
    ax, Sigma, mu, xlim, ylim,
    n=400,
    color_px="C0", color_py="C1", color_mean="C2",
):
    tmin = min(xlim[0], ylim[0])
    tmax = max(xlim[1], ylim[1])
    t = np.linspace(tmin, tmax, n)

    px = _gauss_pdf_1d(t, mu[0], Sigma[0, 0])
    py = _gauss_pdf_1d(t, mu[1], Sigma[1, 1])
    pmean = 0.5 * (px + py)

    ax.plot(t, px, lw=2., color=color_px, label=r"$p(x)$")
    ax.plot(t, py, lw=2., color=color_py, label=r"$p(y)$")
    ax.plot(t, pmean, lw=2., ls="--", color=color_mean, label=r"Mean")

    # mark the means (same colors)
    # ax.axvline(mu[0], ls=":", lw=1.4, color=color_px)
    # ax.axvline(mu[1], ls=":", lw=1.4, color=color_py)

    ax.set_xlim(tmin, tmax)
    # ax.set_ylim(0., max(px.max(), py.max()) * 1.05)

    ax.set_ylim(0., 1.5)
    ax.set_yticks(np.linspace(0., 1.5, 6))

    ax.set_xlabel("Value", fontproperties=font)
    ax.set_ylabel(r"$p$", rotation=0., fontproperties=font)
    ax.yaxis.set_label_coords(-.05, .5)
    # ax.legend(framealpha=1., prop=font_legend)

    ax.set_xticks(t[::max(1, n // 20 * 5)])
    ax.grid()
    ax.tick_params(axis='both', which='both',
                   bottom=False, top=False, left=False, right=False,
                   labelbottom=False, labelleft=False)


def plot_gaussian_2x2_with_rotation(
):
    Sigma = np.array([[.1, .02], [.02, .5]], dtype=float)
    xlim=(-2., 2.)
    ylim=(-2., 2.)
    colors=("dodgerblue", "red", "black")  # (p(x), p(y), mean)

    mu = np.asarray((0., 0.,), dtype=float).reshape(2,)

    R = _rotmat(45.)
    Sigma_rot = R @ Sigma @ R.T

    c_px, c_py, c_mean = colors

    fig = plt.figure(figsize=(6., 6.))
    gs = fig.add_gridspec(2, 2)

    ax00 = fig.add_subplot(gs[0, 0], projection="3d")
    ax01 = fig.add_subplot(gs[0, 1])
    ax10 = fig.add_subplot(gs[1, 0], projection="3d")
    ax11 = fig.add_subplot(gs[1, 1])

    _plot_joint_3d(
        ax00, Sigma, mu, xlim, ylim,
        color_px=c_px, color_py=c_py,
        add_ellipse_contours=True,
    )
    _plot_marginals_2d(ax01, Sigma, mu, xlim, ylim, color_px=c_px, color_py=c_py, color_mean=c_mean)

    _plot_joint_3d(
        ax10, Sigma_rot, mu, xlim, ylim,
        color_px=c_px, color_py=c_py,
        add_ellipse_contours=True,
    )
    _plot_marginals_2d(ax11, Sigma_rot, mu, xlim, ylim, color_px=c_px, color_py=c_py, color_mean=c_mean)

    ax00.set_title("Joint Distribution (Original)", fontproperties=font)
    ax01.set_title("Marginal Distribution (Original)", fontproperties=font)
    ax10.set_title(f"Joint Distribution (Rotated)", fontproperties=font)
    ax11.set_title("Marginal Distribution (Rotated)", fontproperties=font)

    ax = ax11
    handles = {}
    handles["$p(x,y)$"] = CmapLegend(mpl.cm.Greens, mpl.colors.TwoSlopeNorm(vmin=-2., vcenter=0., vmax=2.), n=80, orientation='horizontal')
    handles["$p(x)$"], = ax.plot(100., 100., color=c_px, linestyle='-', linewidth=2.)
    handles["$p(y)$"], = ax.plot(100., 100., color=c_py, linestyle='-', linewidth=2.)
    handles["$p(x)$, $p(y)$ Mean"], = ax.plot(100., 100., color=c_mean, linestyle='--', linewidth=2.)
    # handles["$p(x,y)$ Contour"] = ax.scatter(100., 100., alpha=1., facecolors='none', edgecolors='gray', s=81., marker='o', linewidths=1.)

    fig.legend(handles.values(), handles.keys(), loc='center left', ncol=1, framealpha=1., bbox_to_anchor=(1., .5), prop=font_legend, handler_map={CmapLegend: HandlerCmap()})
    fig.set_facecolor((1., 1., 1., 0.))
    fig.tight_layout()
    fig.savefig('4_marginals.pdf', bbox_inches='tight', pad_inches=.01, transparent=False, dpi=300, metadata=strip_pdf_meta)
    fig.savefig('4_marginals.svg', bbox_inches='tight', pad_inches=.01, transparent=False, dpi=300, metadata=strip_svg_meta)
    fig.show()


if __name__ == '__main__':
    plot_gaussian_2x2_with_rotation()
