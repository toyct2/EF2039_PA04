"""
Step 6. Evaluate RQ2: cross-domain test (2 models x 2 test sets).

Models (Teachable Machine Keras exports):
    mono model: models/rq1/11_324_N1_D1          (reused from RQ1)
    poly model: models/rq2/poly                  (trained on dataset/rq2/train_poly_324_N1-D1)

Output:
    results/rq2/rq2_results.csv
    results/rq2/<train>_to_<test>/wrong/   misclassified test images

Usage (run from the pa04 folder, in the conda environment):
    python scripts/06_evaluate_rq2.py
"""
import csv
import importlib.util
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "results" / "rq2"

MODELS = {
    "mono": ROOT / "models" / "rq1" / "11_324_N1_D1",
    "poly": ROOT / "models" / "rq2" / "poly",
}
TEST_SETS = {
    "mono": ROOT / "dataset" / "rq2" / "test_mono",
    "poly": ROOT / "dataset" / "rq2" / "test_poly",
}

# Reuse loading, preprocessing and metric functions from the RQ1 evaluation script
_spec = importlib.util.spec_from_file_location("eval_rq1", Path(__file__).parent / "03_evaluate_rq1.py")
rq1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rq1)


def load_test_set(test_dir):
    paths, labels = [], []
    for label in rq1.CLASSES:
        for fname in sorted(os.listdir(test_dir / label)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(test_dir / label / fname)
                labels.append(label)
    return paths, labels, np.stack([rq1.preprocess(p) for p in paths])


def save_wrong(folder, paths, true_labels, pred_labels, scores):
    out = RESULT_DIR / folder / "wrong"
    out.mkdir(parents=True, exist_ok=True)
    for path, t, p, s in zip(paths, true_labels, pred_labels, scores):
        if t != p:
            image = rq1.imread(path)
            cv2.putText(image, f"GT:{t} Pred:{p} {s * 100:.0f}%", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            rq1.imwrite(out / path.name, image)


def main():
    test_data = {name: load_test_set(d) for name, d in TEST_SETS.items()}

    results = []
    for train_name, model_dir in MODELS.items():
        model, class_names = rq1.load_tm_model(model_dir)
        for test_name, (paths, true_labels, images) in test_data.items():
            prediction = model.predict(images, verbose=0)
            pred_labels = [class_names[i] for i in prediction.argmax(axis=1)]
            m = rq1.compute_metrics(true_labels, pred_labels)
            save_wrong(f"{train_name}_to_{test_name}", paths, true_labels,
                       pred_labels, prediction.max(axis=1))

            results.append({"train": train_name, "test": test_name, **m})
            print(f"train={train_name:<5} test={test_name:<5} "
                  f"normal={m['normal_acc']*100:6.2f}%  defect={m['defect_acc']*100:6.2f}%  "
                  f"balanced={m['balanced_acc']*100:6.2f}%  F1={m['f1']:.3f}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULT_DIR / "rq2_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        for r in results:
            writer.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
    print(f"\nSaved: {RESULT_DIR / 'rq2_results.csv'}")


if __name__ == "__main__":
    main()
