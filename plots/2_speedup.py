from matplotlib import pyplot as plt
import numpy as np

plt.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'path',
    'svg.hashsalt': 'fixed-salt',  # any constant string for reproducibility
})
strip_svg_meta: dict[str, None] = {k: None for k in ('Creator', 'Date', 'Format', 'Type')}
strip_pdf_meta: dict[str, None] = {k: None for k in ('Title', 'Author', 'Subject', 'Keywords', 'Creator', 'Producer', 'CreationDate', 'ModDate', 'Trapped')}


def plot_speedup() -> None:
    # =========================
    # BS = 1024 (ALL MODELS)
    # =========================
    # Per-layer FP4-MatMul speedups vs. BF16 at batch size M=1024 (from the kernel benchmarks).

    # Qwen3-8B @ BS=1024
    q8_labels = ["N=4096\nK=4096", "N=24576\nK=4096", "N=4096\nK=12288"]
    q8_noquant = np.array([6.210, 5.461, 6.628])
    q8_had     = np.array([4.940, 5.179, 5.520])
    q8_wush    = np.array([4.852, 5.166, 5.397])

    # Qwen3-14B @ BS=1024
    q14_labels = ["N=5120\nK=5120", "N=34816\nK=5120", "N=5120\nK=17408"]
    q14_noquant = np.array([5.863, 5.276, 6.030])
    q14_had     = np.array([4.874, 5.136, 4.900])
    q14_wush    = np.array([4.775, 5.123, 4.756])

    # Llama-3.1-70B @ BS=1024
    l70_labels = ["N=8192\nK=8192", "N=57344\nK=8192", "N=8192\nK=28672"]
    l70_noquant = np.array([6.441, 5.246, 6.219])
    l70_had     = np.array([5.794, 5.141, 5.581])
    l70_wush    = np.array([5.756, 5.135, 5.497])

    models = [
        ("Qwen3-8B", q8_labels, q8_noquant, q8_had, q8_wush),
        ("Qwen3-14B", q14_labels, q14_noquant, q14_had, q14_wush),
        ("Llama-3.1-70B", l70_labels, l70_noquant, l70_had, l70_wush),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(9., 4.), sharey=True)
    width = 0.24

    for ax, (title, labels, noq, had, wush) in zip(axes, models):
        x = np.arange(len(labels))
        ax.bar(x - width, noq, width, label="None + None + FP4-MatMul", facecolor='dodgerblue', edgecolor="black", linewidth=0.3)
        ax.bar(x, had, width, label="H + Quant + FP4-MatMul", facecolor='red', edgecolor="black", linewidth=0.3)
        ax.bar(x + width, wush, width, label="WUSH + Quant + FP4-MatMul", facecolor='forestgreen', edgecolor="black", linewidth=0.3)
        ax.axhline(1.0, linestyle="--", color="black", linewidth=1.0)

        ax.set_title(title, fontsize=15)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylim(0., 7.)
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='both', which='both', length=0)
        ax.grid(True, axis="y", linewidth=0.4, alpha=1.)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Speedup vs. BF16 (M=1024)", fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .0), ncols=3, fontsize=10, frameon=True, framealpha=1.)
    fig.set_facecolor((1., 1., 1., 0.))

    fig.tight_layout()
    fig.savefig('2_speedup.pdf', bbox_inches='tight', pad_inches=.01, transparent=False, metadata=strip_pdf_meta)
    fig.savefig('2_speedup.svg', bbox_inches='tight', pad_inches=.01, transparent=False, metadata=strip_svg_meta)
    fig.show()


if __name__ == '__main__':
    plot_speedup()
