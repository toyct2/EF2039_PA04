"""
Step 4. Plot RQ1 results (figure for the paper).

Input:  results/rq1/rq1_results.csv   (from 03_evaluate_rq1.py)
Output: results/rq1/fig_rq1.pdf       (vector, for LaTeX/Overleaf)
        results/rq1/fig_rq1.png       (preview)

Usage (run from the pa04 folder, in the conda environment):
    python scripts/04_plot_rq1.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "results" / "rq1"

# x-axis: share of normal images increases left -> right
RATIOS = ["1:3", "1:2", "1:1", "2:1", "3:1"]
SIZES = [36, 108, 324]
COLORS = {36: "#86b6ef", 108: "#2a78d6", 324: "#0d366b"}   # light -> dark = small -> large
MARKERS = {36: "o", 108: "s", 324: "^"}

PANELS = [
    ("defect_acc", "(a) Defect recall"),
    ("normal_acc", "(b) Normal accuracy"),
    ("balanced_acc", "(c) Balanced accuracy"),
]


def load_results():
    with open(RESULT_DIR / "rq1_results.csv") as f:
        rows = list(csv.DictReader(f))
    table = {}
    for r in rows:
        table[(int(r["size"]), r["ratio"])] = {k: float(r[k]) * 100 for k, _ in PANELS}
    return table


def main():
    table = load_results()

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2), sharey=True)

    for ax, (metric, title) in zip(axes, PANELS):
        for size in SIZES:
            y = [table[(size, ratio)][metric] for ratio in RATIOS]
            ax.plot(RATIOS, y, color=COLORS[size], marker=MARKERS[size],
                    markersize=5, linewidth=1.5, label=f"{size} images")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Train ratio (normal : defect)")
        ax.set_ylim(55, 100)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
        ax.tick_params(length=2)

    axes[0].set_ylabel("Score on mono test set (%)")
    axes[0].legend(title="Train size", frameon=False, fontsize=7, title_fontsize=7,
                   loc="lower left")

    fig.tight_layout()
    fig.savefig(RESULT_DIR / "fig_rq1.pdf")
    fig.savefig(RESULT_DIR / "fig_rq1.png", dpi=300)
    print(f"Saved: {RESULT_DIR / 'fig_rq1.pdf'}")


if __name__ == "__main__":
    main()
