import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import numpy as np

plt.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'path',
    'svg.hashsalt': 'fixed-salt',  # any constant string for reproducibility
})
strip_svg_meta: dict[str, None] = {k: None for k in ('Creator', 'Date', 'Format', 'Type')}
strip_pdf_meta: dict[str, None] = {k: None for k in ('Title', 'Author', 'Subject', 'Keywords', 'Creator', 'Producer', 'CreationDate', 'ModDate', 'Trapped')}

font = mpl.font_manager.FontProperties(size=11)
font_legend = mpl.font_manager.FontProperties(size=11)


def plot_2d() -> None:
    colors = 'dodgerblue', 'red', 'forestgreen'

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4., 4.))

    x_lim = -.5, 1.5
    y_lim = .5, 2.5
    n_ticks = 5
    n_samples = 100
    x_long = np.linspace(x_lim[0], x_lim[1], round(n_samples * (x_lim[1] - x_lim[0])))
    x_01 = np.linspace(0., 1., n_samples)

    l1__, = ax.plot(x_long, 1. + x_long, color=colors[0], linestyle=':', linewidth=1., zorder=2.2)
    l2__, = ax.plot(x_long, 2. ** x_long, color=colors[1], linestyle=':', linewidth=1., zorder=2.2)
    l1_, = ax.plot(x_01, 1. + x_01, color=colors[0], linestyle='-', linewidth=2., zorder=2.2)
    l2_, = ax.plot(x_01, 2. ** x_01, color=colors[1], linestyle='-', linewidth=2., zorder=2.2)
    x_emax = -np.log2(np.log(2))  # point of maximum gap between y=1+x and y=2^x on [0, 1]
    le, = ax.plot([x_emax, x_emax], [2. ** x_emax, 1 + x_emax], color=colors[2], linestyle='-', linewidth=1.5, zorder=2.1)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_xticks(np.linspace(x_lim[0], x_lim[1], n_ticks))
    ax.set_yticks(np.linspace(y_lim[0], y_lim[1], n_ticks))
    ax.set_xticklabels([f'{tick}' for tick in ax.get_xticks()], fontproperties=font)
    ax.set_yticklabels([f'{tick}' for tick in ax.get_yticks()], fontproperties=font)
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xlabel(r'$x$', rotation=0., fontproperties=font)
    ax.set_ylabel(r'$y$', rotation=0., fontproperties=font)
    ax.yaxis.set_label_coords(-.12, .5)
    ax.set_aspect('equal')
    ax.grid()
    ax.legend([(l1_, l1__), (l2_, l2__)], [r'$y=1+x$', r'$y=2^x$'], framealpha=1., prop=font_legend, handler_map={tuple: HandlerTuple(ndivide=None)})
    ax.text(x_emax - .01, (1. + x_emax + 2. ** x_emax) * .5 - .11, f'max difference ≈ {(1. + x_emax) - 2. ** x_emax:.3f}', color=colors[2], ha='left', va='center', fontproperties=mpl.font_manager.FontProperties(size=9))
    ax.set_facecolor((1., 1., 1., 1.))
    fig.set_facecolor((1., 1., 1., 0.))
    fig.tight_layout()
    fig.savefig('6_fp_model.pdf', bbox_inches='tight', pad_inches=.01, transparent=False, metadata=strip_pdf_meta)
    fig.savefig('6_fp_model.svg', bbox_inches='tight', pad_inches=.01, transparent=False, metadata=strip_svg_meta)
    fig.show()
    # fig.clf()


if __name__ == '__main__':
    plot_2d()
