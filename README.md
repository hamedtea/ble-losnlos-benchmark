# BLE LOS/NLOS Benchmark

This repository contains the datasets, source code, trained machine learning models, and evaluation scripts accompanying the paper

> **Lightweight Non-Line-of-Sight Channel Detection for ML-assisted Bluetooth Direction Finding**

## Overview

Bluetooth Low Energy (BLE) Direction Finding (DF) enables low-cost indoor localization but suffers from significant performance degradation under non-line-of-sight (NLOS) propagation. This repository provides a benchmark framework for BLE LOS/NLOS channel detection using Constant Tone Extension (CTE) IQ measurements and lightweight machine learning pipelines.

The repository includes

- BLE LOS/NLOS datasets collected in controlled indoor environments 
- Feature engineering and preprocessing pipelines
- Nyström Kernel Approximation (NKA) and Random Fourier Features (RFF)
- Principal Component Analysis (PCA)
- Adaptive Kernel Density Estimation (AKDE)
- Trained classifiers
  - Support Vector Classifier (SVC)
  - Random Forest (RF)
  - Multi-layer Perceptron (MLP)
- Evaluation scripts reproducing the paper results

--- The original IQ samples cannot be distributed due to a non-disclosure agreement (NDA). Nevertheless, the provided code can be executed and tested using the included example data or user-provided datasets.

## Repository Structure

```text
ble-losnlos-benchmark/
│
├── data/
│   ├── room/
│   ├── office/
│   ├── mixed/
│   └── proportional/
│
├── models/
│   ├── saved_models/
│   └── results_3k_7k.joblib
│
├── notebooks/
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── classifiers.py
│   ├── evaluation.py
│   └── utils.py
│
├── figures/
│
├── results/
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Datasets

Four datasets are included.

| Dataset | Description |
|---------|-------------|
| Room | Controlled room environment |
| Office | Office environment |
| Mixed | Combination of Room and Office |
| Proportional | Balanced Room–Office dataset |

Each observation corresponds to one BLE CTE packet represented by IQ-derived features together with the corresponding LOS/NLOS label.

---

## Machine Learning Pipelines

The benchmark evaluates multiple feature-engineering pipelines followed by lightweight classifiers.

### Feature Engineering

- Robust Scaling
- PCA
- Nyström Kernel Approximation
- Random Fourier Features
- Cosine Kernel
- Polynomial Kernel
- Sigmoid Kernel
- Linear Projection

### Classifiers

- Linear Support Vector Classifier
- Random Forest
- Multi-layer Perceptron

---

## Installation

Clone the repository

```bash
git clone https://github.com/hamedtea/ble-losnlos-benchmark.git
cd ble-losnlos-benchmark
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Loading Trained Models

```python
from joblib import load

model = load("models/saved_models/RFF_plus_SVC.joblib")

y_pred = model.predict(X)

y_prob = model.predict_proba(X)
```

---

## Reproducing the Paper

The complete benchmark can be reproduced by

1. Loading one of the datasets.
2. Applying the corresponding preprocessing pipeline.
3. Loading the trained model.
4. Running the evaluation scripts.

Performance metrics include

- Accuracy
- TPR
- TNR
- F1-score
- Precision
- Recall
- Cohen's κ
- ROC-AUC
- Training time
- Inference time

---

## Citation

If you use this repository, please cite

```bibtex
@article{talebian2026ble,
  title={Lightweight Non-Line-of-Sight Channel Detection for ML-assisted Bluetooth Direction Finding},
  author={Talebian, Hamed and Mahmood, Aamir and Haghshenas, Mehdi and Rydbloom, Stefani and Karlsson, Peter and Gidlund, Mikael},
  Conference={IPIN},
  year={2026}
}
```

---

## License

This repository is released under the MIT License unless otherwise stated.

---

## Contact

**Hamed Talebian**

Department of Computer and Electrical Engineering

Mid Sweden University

Email: hamed.talebian@miun.se