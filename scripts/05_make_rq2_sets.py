"""
Step 5. Build datasets for RQ2 (cross-domain: mono <-> poly).

RQ2: Does a model trained on one cell type (mono or poly) generalize to the other?

- Mono model: reuse RQ1 model 11 (mono, 324 images, normal:defect = 1:1).
- Poly model: needs a poly training set with the same size and ratio
  -> 162 normal + 162 defect poly cells, sampled with a fixed seed.
- Test sets: all mono test cells and all poly test cells (never used for training).

Output:
    dataset/rq2/train_poly_324_N1-D1/{normal,defect}/
    dataset/rq2/test_mono/{normal,defect}/
    dataset/rq2/test_poly/{normal,defect}/
    dataset/rq2/summary.csv

Usage (run from the pa04 folder, after 01_make_split.py):
    python scripts/05_make_rq2_sets.py
"""
import csv
import random
import shutil
from pathlib import Path

SEED = 42
N_NORMAL, N_DEFECT = 162, 162            # same as the mono model (RQ1 model 11)

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
OUT = DATASET / "rq2"


def read_split(name, mtype):
    with open(DATASET / f"{name}.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["type"] == mtype]
    normal = sorted(r["file"] for r in rows if r["label"] == "normal")
    defect = sorted(r["file"] for r in rows if r["label"] == "defect")
    return normal, defect


def copy_set(folder, split, normal, defect):
    for label, files in [("normal", normal), ("defect", defect)]:
        dst = OUT / folder / label
        dst.mkdir(parents=True, exist_ok=True)
        for fname in files:
            shutil.copy(DATASET / split / label / fname, dst / fname)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)

    normal_pool, defect_pool = read_split("train", "poly")
    rng = random.Random(SEED)
    rng.shuffle(normal_pool)
    rng.shuffle(defect_pool)
    copy_set("train_poly_324_N1-D1", "train", normal_pool[:N_NORMAL], defect_pool[:N_DEFECT])
    summary = [["train_poly_324_N1-D1", N_NORMAL, N_DEFECT]]

    for mtype in ["mono", "poly"]:
        normal, defect = read_split("test", mtype)
        copy_set(f"test_{mtype}", "test", normal, defect)
        summary.append([f"test_{mtype}", len(normal), len(defect)])

    with open(OUT / "summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["folder", "normal", "defect"])
        writer.writerows(summary)

    for name, n, d in summary:
        print(f"{name:<24} total={n + d:>3}  normal={n:>3}  defect={d:>3}")


if __name__ == "__main__":
    main()
