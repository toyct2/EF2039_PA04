# Data Effects on Solar Cell Defect Classification (EF2039 PA04)

Binary classification of electroluminescence (EL) images of solar cells (normal vs. defect).
We study two data-related questions:

- **RQ1.** How do training-set size and class ratio affect defect classification? (mono cells)
- **RQ2.** Does a model trained on one cell type (mono or poly) generalize to the other?

## Repository structure

```
pa04/
├── environment.yml          # conda environment (pinned versions)
├── scripts/
│   ├── 01_make_split.py     # filter labels, 80/20 stratified train/test split
│   ├── 02_make_rq1_sets.py  # 15 training sets for RQ1 + mono test set
│   ├── 03_evaluate_rq1.py   # evaluate the 15 RQ1 models
│   ├── 04_plot_rq1.py       # RQ1 figure
│   ├── 05_make_rq2_sets.py  # poly training set + mono/poly test sets for RQ2
│   ├── 06_evaluate_rq2.py   # evaluate the 2 RQ2 models on both test sets
│   └── 07_make_rq2_figures.py  # RQ2 table + sample image figure
├── dataset/                 # split lists (CSV); image folders are rebuilt by the scripts
├── models/                  # trained Teachable Machine models (Keras .h5)
│   ├── rq1/<id>_<size>_N<normal>_D<defect>/
│   └── rq2/poly/
└── results/                 # metrics (CSV), figures, LaTeX table
```

## 1. Setup

Tested on Windows 11 with Miniconda, CPU only (no GPU needed; evaluation takes a few minutes).

```
git clone https://github.com/<your-id>/<repo-name>.git
cd <repo-name>
conda env create -f environment.yml
conda activate pa04
```

All package versions are pinned in `environment.yml` (Python 3.10, TensorFlow 2.15.0),
so the environment does not depend on the conda solver at install time.

## 2. Data

We use the ELPV dataset (2,624 EL images of solar cells, 300×300 grayscale, CC BY-NC-SA 4.0).

1. Download the ZIP from https://github.com/zae-bayern/elpv-dataset (Code → Download ZIP).
2. Extract it into the repository root so the folder is named `elpv-dataset-master/`.

Preprocessing and splits:

- Each image has a defect probability (0, 1/3, 2/3, 1). Only certain labels are kept:
  0 → `normal`, 1 → `defect`. This leaves 2,223 images (1,508 normal, 715 defect).
- 80/20 split, stratified by (label, cell type), seed 42 → 1,778 train / 445 test.
- The exact file lists are in `dataset/train.csv`, `dataset/test.csv`,
  `dataset/rq1/summary.csv`, `dataset/rq2/summary.csv`.

Rebuild all image folders:

```
python scripts/01_make_split.py
python scripts/02_make_rq1_sets.py
python scripts/05_make_rq2_sets.py
```

| Set | Cell type | Normal | Defect |
|---|---|---|---|
| RQ1 training sets (15) | mono | 3 sizes (36, 108, 324) × 5 ratios (1:3 … 3:1) | |
| RQ2 training set | poly | 162 | 162 |
| Test set | mono | 118 | 63 |
| Test set | poly | 184 | 80 |

Smaller RQ1 sets are subsets of larger sets with the same ratio.

## 3. Training (Teachable Machine)

Trained models are already included in `models/`, so this step can be skipped.
To retrain, use https://teachablemachine.withgoogle.com → Image Project → Standard image model:

1. Rename Class 1 to `normal` and Class 2 to `defect` (this order), and upload the
   images from the matching subfolders of a training set.
2. Keep the default settings: **epochs 50, batch size 16, learning rate 0.001**.
   (Teachable Machine internally holds out 15% of the uploaded images for its own validation.)
3. Export: Tensorflow → Keras → Download, and extract `keras_model.h5` and `labels.txt`.

| Training set folder | Model folder |
|---|---|
| `dataset/rq1/train_size036_N1-D1` | `models/rq1/01_36_N1_D1` |
| … (same pattern, in folder name order) | `models/rq1/02_36_N1_D2` … `15_324_N3_D1` |
| `dataset/rq2/train_poly_324_N1-D1` | `models/rq2/poly` |

The mono model for RQ2 is RQ1 model `11_324_N1_D1` (324 images, 1:1).

## 4. Evaluation and figures

```
python scripts/03_evaluate_rq1.py
python scripts/04_plot_rq1.py
python scripts/06_evaluate_rq2.py
python scripts/07_make_rq2_figures.py
```

Preprocessing at test time follows the Teachable Machine export: resize to 224×224, scale to [-1, 1].
The positive class is `defect`. Reported metrics: normal accuracy (specificity),
defect recall, overall accuracy, balanced accuracy, precision, F1.

Outputs: `results/rq1/rq1_results.csv`, `results/rq1/fig_rq1.pdf`,
`results/rq2/rq2_results.csv`, `results/rq2/table_rq2.tex`, `results/rq2/fig_samples.pdf`.

## 5. Expected results

RQ1 (mono test set, %):

| Model | Size | Normal:Defect | Normal acc. | Defect recall | Balanced acc. |
|---|---|---|---|---|---|
| 01_36_N1_D1 | 36 | 1:1 | 85.6 | 81.0 | 83.3 |
| 02_36_N1_D2 | 36 | 1:2 | 77.1 | 85.7 | 81.4 |
| 03_36_N1_D3 | 36 | 1:3 | 66.1 | 90.5 | 78.3 |
| 04_36_N2_D1 | 36 | 2:1 | 91.5 | 73.0 | 82.3 |
| 05_36_N3_D1 | 36 | 3:1 | 97.5 | 68.2 | 82.9 |
| 06_108_N1_D1 | 108 | 1:1 | 83.9 | 79.4 | 81.6 |
| 07_108_N1_D2 | 108 | 1:2 | 73.7 | 90.5 | 82.1 |
| 08_108_N1_D3 | 108 | 1:3 | 75.4 | 87.3 | 81.4 |
| 09_108_N2_D1 | 108 | 2:1 | 91.5 | 65.1 | 78.3 |
| 10_108_N3_D1 | 108 | 3:1 | 94.9 | 60.3 | 77.6 |
| 11_324_N1_D1 | 324 | 1:1 | 93.2 | 79.4 | 86.3 |
| 12_324_N1_D2 | 324 | 1:2 | 77.1 | 93.7 | 85.4 |
| 13_324_N1_D3 | 324 | 1:3 | 67.8 | 93.7 | 80.7 |
| 14_324_N2_D1 | 324 | 2:1 | 92.4 | 73.0 | 82.7 |
| 15_324_N3_D1 | 324 | 3:1 | 94.9 | 69.8 | 82.4 |

RQ2 (%):

| Train | Test | Normal acc. | Defect recall | Balanced acc. |
|---|---|---|---|---|
| Mono | Mono | 93.2 | 79.4 | 86.3 |
| Mono | Poly | 8.2 | 96.2 | 52.2 |
| Poly | Mono | 20.3 | 82.5 | 51.4 |
| Poly | Poly | 77.2 | 75.0 | 76.1 |

Evaluation of the included models is deterministic; retrained models will give slightly different numbers.

## Acknowledgements

Dataset: Buerhop-Lutz et al., EU PVSEC 2018; Deitsch et al., Solar Energy 2019.
Evaluation code is based on the course example `03_evaluate_ai_model.py` (EF2039, KENTECH).
