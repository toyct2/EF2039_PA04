"""
Step 3. Evaluate all RQ1 models on the fixed mono test set.
(Based on the course code 03_evaluate_ai_model.py)

Models: every folder under pa04/ that contains keras_model.h5 + labels.txt
is evaluated (one Teachable Machine export per folder; folder name = model name).
Training size and class ratio are read from MODEL_INFO, or parsed from
folder names like "01_36_N1_D1" (index_size_N<normal>_D<defect>).

Output:
    results/rq1/rq1_results.csv              metrics for every model
    results/rq1/<model>/{correct,wrong}/     test images with predictions drawn

Usage (run from the pa04 folder, in the conda environment):
    python scripts/03_evaluate_rq1.py
"""
import csv
import io
import os
import re
from pathlib import Path

import cv2
import h5py
import numpy as np
from tensorflow.keras.models import load_model

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "dataset" / "rq1" / "test_mono"
RESULT_DIR = ROOT / "results" / "rq1"

CLASSES = ["normal", "defect"]   # must match Teachable Machine class names
POSITIVE = "defect"              # defect is the positive class for precision/recall

# Model folder name -> (training size, normal:defect ratio)
MODEL_INFO = {
    # "my_model_folder": (36, "1:1"),
}


def find_model_folders():
    """Every folder under pa04/ that holds a Teachable Machine Keras export."""
    return sorted({p.parent for p in ROOT.rglob("*")
                   if p.name.lower() == "keras_model.h5"
                   and (p.parent / "labels.txt").exists()})


def model_info(name):
    if name in MODEL_INFO:
        return MODEL_INFO[name]
    # e.g. "01_36_N1_D1" or "train_size036_N1-D1" -> (36, "1:1")
    m = re.search(r"(\d+)_N(\d+)[-_]D(\d+)", name)
    if m:
        return int(m.group(1)), f"{m.group(2)}:{m.group(3)}"
    return "", ""


def imread(path):
    # cv2.imread fails on Windows paths with non-ASCII (e.g. Korean) characters
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite(path, image):
    cv2.imencode(Path(path).suffix, image)[1].tofile(str(path))


def preprocess(img_path):
    """Same preprocessing as the course code: resize to 224x224, scale to [-1, 1]."""
    image = imread(img_path)                           # grayscale PNG -> 3 equal channels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    return np.asarray(image, dtype=np.float32) / 127.5 - 1


def load_test_set():
    paths, labels = [], []
    for label in CLASSES:
        for fname in sorted(os.listdir(TEST_DIR / label)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(TEST_DIR / label / fname)
                labels.append(label)
    images = np.stack([preprocess(p) for p in paths])
    return paths, labels, images


def load_tm_model(folder):
    h5 = [f for f in os.listdir(folder) if f.lower() == "keras_model.h5"][0]
    # TensorFlow's file I/O breaks on Windows paths with Korean characters,
    # so read the .h5 into memory and hand Keras an in-memory HDF5 file.
    with open(folder / h5, "rb") as f:
        h5_file = h5py.File(io.BytesIO(f.read()), "r")
    model = load_model(h5_file, compile=False)
    with open(folder / "labels.txt") as f:
        class_names = [line[2:].strip() for line in f]   # "0 normal" -> "normal"
    return model, class_names


def compute_metrics(true_labels, pred_labels):
    tp = sum(t == POSITIVE and p == POSITIVE for t, p in zip(true_labels, pred_labels))
    tn = sum(t != POSITIVE and p != POSITIVE for t, p in zip(true_labels, pred_labels))
    fp = sum(t != POSITIVE and p == POSITIVE for t, p in zip(true_labels, pred_labels))
    fn = sum(t == POSITIVE and p != POSITIVE for t, p in zip(true_labels, pred_labels))

    normal_acc = tn / (tn + fp)                       # specificity
    defect_acc = tp / (tp + fn)                       # recall
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * defect_acc / (precision + defect_acc) if precision + defect_acc else 0.0
    return {
        "normal_acc": normal_acc,
        "defect_acc": defect_acc,
        "overall_acc": (tp + tn) / len(true_labels),
        "balanced_acc": (normal_acc + defect_acc) / 2,
        "precision": precision,
        "f1": f1,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }


def save_images(model_name, paths, true_labels, pred_labels, scores):
    for sub in ["correct", "wrong"]:
        (RESULT_DIR / model_name / sub).mkdir(parents=True, exist_ok=True)
    for path, t, p, s in zip(paths, true_labels, pred_labels, scores):
        image = imread(path)
        cv2.putText(image, f"GT:{t} Pred:{p} {s * 100:.0f}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        sub = "correct" if t == p else "wrong"
        imwrite(RESULT_DIR / model_name / sub / path.name, image)


def main():
    os.chdir(ROOT)
    paths, true_labels, images = load_test_set()
    print(f"Test set: {len(paths)} images "
          f"(normal={true_labels.count('normal')}, defect={true_labels.count('defect')})")

    results = []
    folders = find_model_folders()
    print(f"Found {len(folders)} models")
    for folder in folders:
        model, class_names = load_tm_model(folder)
        assert sorted(class_names) == sorted(CLASSES), \
            f"{folder.name}: labels.txt has {class_names}, expected {CLASSES}"

        prediction = model.predict(images, verbose=0)
        pred_labels = [class_names[i] for i in prediction.argmax(axis=1)]
        scores = prediction.max(axis=1)

        m = compute_metrics(true_labels, pred_labels)
        save_images(folder.name, paths, true_labels, pred_labels, scores)

        size, ratio = model_info(folder.name)
        results.append({"model": folder.name, "size": size, "ratio": ratio, **m})
        print(f"{folder.name:<26} normal={m['normal_acc']*100:6.2f}%  "
              f"defect={m['defect_acc']*100:6.2f}%  overall={m['overall_acc']*100:6.2f}%  "
              f"balanced={m['balanced_acc']*100:6.2f}%  F1={m['f1']:.3f}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULT_DIR / "rq1_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        for r in results:
            writer.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
    print(f"\nSaved: {RESULT_DIR / 'rq1_results.csv'}")


if __name__ == "__main__":
    main()
