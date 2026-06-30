import os

import matplotlib as mpl
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

plt.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'path',
    'svg.hashsalt': 'fixed-salt',  # any constant string for reproducibility
})
strip_svg_meta: dict[str, None] = {k: None for k in ('Creator', 'Date', 'Format', 'Type')}
strip_pdf_meta: dict[str, None] = {k: None for k in ('Title', 'Author', 'Subject', 'Keywords', 'Creator', 'Producer', 'CreationDate', 'ModDate', 'Trapped')}

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, '3_platinum_bench')
DATA_CSV = os.path.join(HERE, '3_platinum_bench.csv')

# (paper figure number, file slug, model name as stored in the CSV)
MODELS = [
    (3, 'qwen3-8b', 'Qwen3-8B'),
    (7, 'qwen3-14b', 'Qwen3-14B'),
    (8, 'qwen3-32b', 'Qwen3-32B'),
    (9, 'llama-3.2-3b', 'Llama-3.2-3B-Instruct'),
    (10, 'llama-3.1-8b', 'Llama-3.1-8B-Instruct'),
]

# The 13 Platinum benchmarks, in the column order used in the paper.
BENCHMARKS = [
    'SingleOp', 'SingleQ', 'MultiArith', 'SVAMP', 'GSM8K', 'MMLU-Math',
    'BBHDeduction', 'BBHCounting', 'BBHNavigate', 'HotpotQA', 'SQuAD',
    'DROP', 'Winograd-WSC',
]

VMIN, VMAX = 80., 100.


def _cell_fmt(x: float) -> str:
    x2 = round(float(x), 2)  # round first (so 99.999 -> 100.0)
    return f'{x2:.1f}' if x2 >= 100. else f'{x2:.2f}'


def _text_color(rgba: tuple[float, ...]) -> str:
    # Pick black/white annotation text by background luminance (matches seaborn).
    def _lin(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    lum = 0.2126 * _lin(rgba[0]) + 0.7152 * _lin(rgba[1]) + 0.0722 * _lin(rgba[2])
    return '0.15' if lum > 0.408 else 'white'


def _save(fig: plt.Figure, prefix: str) -> None:
    fig.set_facecolor((1., 1., 1., 0.))
    fig.tight_layout()
    fig.savefig(prefix + '.pdf', bbox_inches='tight', pad_inches=.01, transparent=False, metadata=strip_pdf_meta)
    fig.savefig(prefix + '.svg', bbox_inches='tight', pad_inches=.01, transparent=False, metadata=strip_svg_meta)
    plt.close(fig)


def plot_table(df: pd.DataFrame, model_name: str, prefix: str) -> None:
    # Per-method mean over benchmarks, with a row-wise Average; baseline -> "BF16".
    mean_df = df.groupby('pretty_name')[BENCHMARKS].mean()
    mean_df['Average'] = mean_df.mean(axis=1)
    mean_df = mean_df.rename(index={f'{model_name}-BF16': 'BF16'})
    mean_df = mean_df.sort_values('Average', ascending=False)

    cols = BENCHMARKS + ['Average']
    data = mean_df[cols].to_numpy()
    nrows, ncols = data.shape

    cmap = mpl.colormaps['Greens']
    norm = mpl.colors.Normalize(VMIN, VMAX)

    fig, ax = plt.subplots(figsize=(18., 10.))
    ax.pcolormesh(data, cmap=cmap, vmin=VMIN, vmax=VMAX, edgecolors='gray', linewidth=0.5)
    ax.invert_yaxis()  # first row at the top, like the table

    for i in range(nrows):
        for j in range(ncols):
            val = data[i, j]
            ax.text(j + .5, i + .5, _cell_fmt(val), ha='center', va='center',
                    color=_text_color(cmap(norm(val))), fontsize=24)

    ax.set_xticks(np.arange(ncols) + .5)
    ax.set_xticklabels(cols, rotation=45, ha='right', fontsize=24)
    ax.set_yticks(np.arange(nrows) + .5)
    ax.set_yticklabels(mean_df.index, rotation=0, fontsize=24)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(f'{model_name} Platinum Benchmarks', fontsize=32, pad=20)
    _save(fig, prefix)


def plot_boxplot(df: pd.DataFrame, model_name: str, prefix: str) -> None:
    data = df.copy()
    data['Average'] = data[BENCHMARKS].mean(axis=1)

    bf16 = f'{model_name}-BF16'
    baseline = data.loc[data['pretty_name'] == bf16, 'Average'].mean()
    data = data[data['pretty_name'] != bf16]

    # Order methods by mean Average (ascending, best on the right).
    order = list(data.groupby('pretty_name')['Average'].mean().sort_values(ascending=False).index)[::-1]
    data_by_method = [data.loc[data['pretty_name'] == m, 'Average'].values for m in order]

    fig, ax = plt.subplots(figsize=(18., 8.))
    bp = ax.boxplot(data_by_method, tick_labels=order, patch_artist=True,
                    flierprops=dict(marker='o', markersize=4, alpha=0.5))
    for box in bp['boxes']:
        box.set_facecolor('dodgerblue')
    for med in bp['medians']:
        med.set_color('black')

    ax.axhline(baseline, linestyle='--', color='red', linewidth=3)

    ax.set_xticklabels(order, rotation=45, ha='right', fontsize=24)
    ax.tick_params(axis='y', labelsize=26)

    ticks = ax.get_yticks()
    ax.set_ylim(ticks[0], ticks[-1])
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

    # "BF16" label just below the baseline line, nudged a few points inward.
    offset = mtransforms.ScaledTranslation(6 / 72, -6 / 72, fig.dpi_scale_trans)
    ax.text(0.5, baseline, 'BF16', transform=ax.transData + offset,
            ha='left', va='top', fontsize=24, color='red')

    ax.set_ylabel('Accuracy [%]', fontsize=32)
    ax.set_title(f'{model_name} Platinum Benchmarks Average Accuracy', fontsize=32)
    ax.grid(True)
    _save(fig, prefix)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_CSV)
    for fig_num, slug, model_name in MODELS:
        sub = df[df['model'] == model_name]
        plot_table(sub, model_name, os.path.join(OUT_DIR, f'{fig_num}_{slug}_table'))
        plot_boxplot(sub, model_name, os.path.join(OUT_DIR, f'{fig_num}_{slug}_boxplot'))


if __name__ == '__main__':
    main()
