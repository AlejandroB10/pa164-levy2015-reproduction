"""Poster chart: vanilla vs recommended setting per method on a few metrics."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

METHODS = ["PPMI", "SVD", "SGNS"]
PANELS = [
    ("ws_sim", "WordSim Similarity\n(Spearman ρ)"),
    ("men", "MEN relatedness\n(Spearman ρ)"),
    ("google_add_30k", "Google analogies, top-30k\n(3CosAdd accuracy)"),
]
VANILLA_COLOR = "#a3a29c"  # neutral: the baseline
TUNED_COLOR = "#2a78d6"  # categorical slot 1: the treatment
WC_COLOR = "#e0752d"  # categorical slot 2: recommended + w+c (SVD and SGNS only)
INK = "#0b0b0b"
MUTED = "#52514e"


def _is_recommended_wc(row):
    """The recommended configuration plus w+c; w+c does not apply to PPMI."""
    return row["role"] == "variant" and row["config"].endswith("w+c") and (
        (row["method"] == "SVD" and "eig=0.5" in row["config"])
        or (row["method"] == "SGNS" and "neg=15" in row["config"])
    )


SERIES = [
    ("vanilla", VANILLA_COLOR, "Vanilla (paper Table 2 defaults)", lambda r: r["role"] == "vanilla"),
    ("recommended", TUNED_COLOR, "Recommended tweaks (paper §7)", lambda r: r["role"] == "recommended"),
    ("recommended_wc", WC_COLOR, "Recommended + w+c", _is_recommended_wc),
]


def poster_chart(results: pd.DataFrame, path, window=2):
    """Grouped bars per method: vanilla, the a-priori recommended configuration, and that plus w+c."""
    data = results[results["win"] == window]
    plt.rcParams.update({"font.size": 18, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, len(PANELS), figsize=(18, 6.8), sharey=False)
    width = 0.27
    for ax, (metric, title) in zip(axes, PANELS):
        sub = data[data["metric"] == metric]
        shown = []
        for k, (_, color, label, select) in enumerate(SERIES):
            offset = (k - 1) * width
            picked = sub[sub.apply(select, axis=1)]
            xs, values = [], []
            for i, m in enumerate(METHODS):
                row = picked[picked["method"] == m]
                if len(row):
                    xs.append(i + offset)
                    values.append(row["value"].iloc[0])
            shown.extend(values)
            bars = ax.bar(xs, values, width=width - 0.03, color=color, label=label, zorder=2)
            for bar, v in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=14, color=INK,
                )
        top = max(shown)
        ax.set_ylim(0, min(1.0, top * 1.22))
        ax.set_xticks(range(len(METHODS)), METHODS, fontsize=20, fontweight="bold", color=INK)
        ax.set_title(title, fontsize=20, color=INK, pad=12)
        ax.tick_params(axis="y", labelsize=14, colors=MUTED)
        ax.grid(axis="y", color="#e4e3df", linewidth=1, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=18,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(
        f"text8 (17M tokens), window {window}: tuning hyperparameters vs switching algorithm",
        y=0.905, fontsize=15, color=MUTED,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
