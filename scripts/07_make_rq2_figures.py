"""
Step 7. Make the RQ2 table and the sample-image figure for the paper.

Input:  results/rq2/rq2_results.csv   (from 06_evaluate_rq2.py)
        dataset/test/{normal,defect}/  (from 01_make_split.py)
Output: results/rq2/table_rq2.tex     LaTeX tabular (use with \\input{} inside a table)
        results/rq2/fig_samples.pdf   2x2 sample EL images (mono/poly x normal/defect)
        results/rq2/fig_samples.png   preview

Usage (run from the pa04 folder, in the conda environment):
    python scripts/07_make_rq2_figures.py
"""
import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "results" / "rq2"
TEST_DIR = ROOT / "dataset" / "test"

# Example test cells. Both normal cells are misclassified as "defect"
# by the model trained on the other cell type (see results/rq2/*/wrong).
SAMPLES = {
    ("Mono", "normal"): "cell0903.png",   # wrong in poly_to_mono
    ("Mono", "defect"): "cell0432.png",
    ("Poly", "normal"): "cell1683.png",   # wrong in mono_to_poly
    ("Poly", "defect"): "cell2042.png",
}


def imread(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def make_table():
    with open(RESULT_DIR / "rq2_results.csv") as f:
        rows = {(r["train"], r["test"]): r for r in csv.DictReader(f)}

    lines = [
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Train & Test & Normal acc. & Defect recall & Balanced acc. \\",
        r"\midrule",
    ]
    for train in ["mono", "poly"]:
        for i, test in enumerate(["mono", "poly"]):
            r = rows[(train, test)]
            bal = f"{float(r['balanced_acc']) * 100:.1f}"
            if train == test:                                  # in-domain result in bold
                bal = rf"\textbf{{{bal}}}"
            lines.append(
                f"{train.capitalize() if i == 0 else ''} & {test.capitalize()} & "
                f"{float(r['normal_acc']) * 100:.1f} & {float(r['defect_acc']) * 100:.1f} & {bal} \\\\"
            )
        if train == "mono":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]

    (RESULT_DIR / "table_rq2.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def make_sample_figure():
    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, axes = plt.subplots(2, 2, figsize=(3.3, 3.5))

    for i, cell_type in enumerate(["Mono", "Poly"]):
        for j, label in enumerate(["normal", "defect"]):
            ax = axes[i, j]
            ax.imshow(imread(TEST_DIR / label / SAMPLES[(cell_type, label)]), cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(label.capitalize(), fontsize=9)
            if j == 0:
                ax.set_ylabel(cell_type, fontsize=9)

    fig.tight_layout(pad=0.3)
    fig.savefig(RESULT_DIR / "fig_samples.pdf")
    fig.savefig(RESULT_DIR / "fig_samples.png", dpi=300)


def main():
    make_table()
    make_sample_figure()
    print(f"\nSaved: {RESULT_DIR / 'table_rq2.tex'}, {RESULT_DIR / 'fig_samples.pdf'}")


if __name__ == "__main__":
    main()
