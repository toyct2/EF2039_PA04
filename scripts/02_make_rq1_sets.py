"""
Step 2. Build training subsets for RQ1 (mono cells only).

RQ1: How do training-set size and class ratio affect defect classification?

- Source: dataset/train.csv and dataset/test.csv from 01_make_split.py
- Mono cells only.
- 3 sizes x 5 ratios (normal:defect) = 15 training sets.
- Smaller sets are subsets of larger sets with the same ratio,
  and all sets draw from the same shuffled pool (fixed seed).
- Test set: all mono test cells (fixed, never used for training).

Output:
    dataset/rq1/train_size036_N1-D1/{normal,defect}/
    ...
    dataset/rq1/test_mono/{normal,defect}/
    dataset/rq1/summary.csv

Usage (run from the pa04 folder, after 01_make_split.py):
    python scripts/02_make_rq1_sets.py
"""
import csv
import random
import shutil
from pathlib import Path

SEED = 42
SIZES = [36, 108, 324]                       # total images per training set
RATIOS = [(1, 1), (2, 1), (3, 1), (1, 2), (1, 3)]   # normal : defect

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
OUT = DATASET / "rq1"


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

    normal_pool, defect_pool = read_split("train", "mono")
    rng = random.Random(SEED)
    rng.shuffle(normal_pool)
    rng.shuffle(defect_pool)

    summary = []
    for n_ratio, d_ratio in RATIOS:
        for size in SIZES:
            n_normal = size * n_ratio // (n_ratio + d_ratio)
            n_defect = size - n_normal
            assert n_normal <= len(normal_pool) and n_defect <= len(defect_pool)
            folder = f"train_size{size:03d}_N{n_ratio}-D{d_ratio}"
            copy_set(folder, "train", normal_pool[:n_normal], defect_pool[:n_defect])
            summary.append([folder, size, f"{n_ratio}:{d_ratio}", n_normal, n_defect])

    test_normal, test_defect = read_split("test", "mono")
    copy_set("test_mono", "test", test_normal, test_defect)
    summary.append(["test_mono", len(test_normal) + len(test_defect), "-",
                    len(test_normal), len(test_defect)])

    with open(OUT / "summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["folder", "total", "normal:defect", "normal", "defect"])
        writer.writerows(summary)

    for row in summary:
        print(f"{row[0]:<28} total={row[1]:>3}  normal={row[3]:>3}  defect={row[4]:>3}")


if __name__ == "__main__":
    main()
