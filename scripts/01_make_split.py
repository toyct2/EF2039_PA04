"""
Step 1. Build train/test split from the ELPV dataset.

- Keep only cells with defect probability 0.0 (normal) or 1.0 (defective).
- Drop ambiguous labels (0.33, 0.67).
- Stratified 80/20 split by (label, module type) with a fixed seed.
- Copy images into class folders: dataset/{train,test}/{normal,defect}/
- Save the exact split as CSV files for reproducibility.

Usage (run from the pa04 folder):
    python scripts/01_make_split.py
"""
import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

SEED = 42
TEST_RATIO = 0.2

ROOT = Path(__file__).resolve().parent.parent          # pa04/
ELPV_DATA = ROOT / "elpv-dataset-master" / "src" / "elpv_dataset" / "data"
OUT = ROOT / "dataset"

LABEL_NAME = {"0.0": "normal", "1.0": "defect"}


def load_labels():
    rows = []
    with open(ELPV_DATA / "labels.csv") as f:
        for line in f:
            path, prob, mtype = line.split()
            if prob in LABEL_NAME:                       # keep only 0 or 1
                rows.append({"path": path, "label": LABEL_NAME[prob], "type": mtype})
    return rows


def stratified_split(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["label"], r["type"])].append(r)

    rng = random.Random(SEED)
    train, test = [], []
    for key in sorted(groups):
        items = groups[key]
        rng.shuffle(items)
        n_test = round(len(items) * TEST_RATIO)
        test += items[:n_test]
        train += items[n_test:]
    return train, test


def save_split(name, rows):
    for label in LABEL_NAME.values():
        (OUT / name / label).mkdir(parents=True, exist_ok=True)

    with open(OUT / f"{name}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "label", "type"])
        writer.writeheader()
        for r in rows:
            src = ELPV_DATA / r["path"]
            fname = Path(r["path"]).name
            shutil.copy(src, OUT / name / r["label"] / fname)
            writer.writerow({"file": fname, "label": r["label"], "type": r["type"]})


def main():
    if OUT.exists():
        shutil.rmtree(OUT)                              # rebuild from scratch

    rows = load_labels()
    train, test = stratified_split(rows)
    save_split("train", train)
    save_split("test", test)

    for name, part in [("train", train), ("test", test)]:
        c = Counter((r["label"], r["type"]) for r in part)
        print(f"[{name}] total={len(part)}  " +
              "  ".join(f"{k[0]}/{k[1]}={v}" for k, v in sorted(c.items())))


if __name__ == "__main__":
    main()
