# Wafer Guard Fixed · Fab-7 Semiconductor Excursion Detector Audit & Repair

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shivam-more1609/GDG_WildCard/blob/main/wafer_guard_fixed.ipynb)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C.svg)](https://pytorch.org/)
[![Release Gate](https://img.shields.io/badge/Release_Gate-REJECTED_(NO--GO)-red.svg)]()
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

Technical audit, root-cause repair, and honest evaluation of the semiconductor inline excursion detection pipeline scheduled for production deployment at Fab-7.

---

## 📋 Executive Summary

The pre-release candidate notebook for the Fab-7 line excursion detector claimed an F1 score of **≥ 0.90** (empirically measured at **0.9915** in our reproduction of the uncorrected pipeline), appearing to clear the release gate. However, our technical audit uncovered **three silent, fatal defects**:
1. **Catastrophic Data Leakage & Artificial Test Prevalence:** Pre-split SMOTE and pre-split mean imputation / scaling leaked future wafer lots and test-set distributions into the training set, while evaluating on an artificially balanced test set (50.00% defect prevalence vs natural ~6.64%).
2. **Corrupted Loss Gradients via Silent Broadcasting:** PyTorch tensor mismatch between reconstructed output `[B, F]` and input tensor `[B, 1, F]` triggered silent broadcasting to `[B, B, F]`, computing cross-sample pairwise distances instead of per-wafer reconstruction MSE.
3. **Invariant Blindness to Legacy Probe Station PS-L:** Hardware sentinel (`sensor_0 == 0.0`) and categorical error codes (`sensor_1 in {100.0, 200.0, 300.0}`) were treated as continuous numbers, corrupting continuous channel scaling and ignoring a 0.00% defect invariant.

Upon repairing all three defects and extracting **595 causal features**, honest evaluation at natural prevalence yielded:
- **Chronological Test F1:** **0.1067** (at 5.41% test set prevalence: 17 defects / 314 lots)
- **Measured Test Precision:** **5.77%** (Recall: 70.59%, False Positive Rate: 65.99%)
- **Prevalence-Adjusted Precision (at Fab Baseline 6.64%):** **7.07%** (Adjusted F1: **0.1285**)
- **Release Gate Verdict:** **REJECTED (NO-GO) ❌**

For the comprehensive technical audit, see [AUDIT.md](AUDIT.md).

---

## 🗂️ Deliverables & Structure

```
wafer-guard-fixed/
├── AUDIT.md                             # Comprehensive technical audit & executive review
├── wafer_guard_fixed.ipynb             # Corrected, fully executed PyTorch notebook
├── README.md                            # Repository documentation & quickstart
├── secom_telemetry_modified.csv         # 1,567 wafer lots with 590 telemetry channels
├── secom_legacy_mask_EVALUATOR_ONLY.csv # Evaluator ground truth for Probe Station PS-L
└── artifacts/                           # Serialized model weights and preprocessor
    ├── wafer_sae_REJECTED_do_not_deploy.pt
    └── preprocessor_REJECTED_do_not_deploy.joblib
```

---

## 📊 Summary of Evaluation Metrics

| Metric | Leaky Candidate (Buggy) | Repaired WaferSAE (Test Split, $\pi=5.41\%$) | Repaired WaferSAE (Adjusted, $\pi=6.64\%$) | Balanced Logistic Regression |
| :--- | :---: | :---: | :---: | :---: |
| **Defect Prevalence ($\pi$)** | 50.00% (synthetic) | **5.41%** (natural holdout) | **6.64%** (fab baseline) | 5.41% |
| **F1 Score** | **0.9915** *(falsified)* | **0.1067** | **0.1285** | 0.1250 |
| **Precision** | 98.32% | **5.77%** | **7.07%** | 6.82% |
| **Recall (Sensitivity)** | 100.00% | **70.59%** | **70.59%** | 64.71% |
| **Specificity** | 98.29% | **34.01%** | **34.01%** | 49.49% |
| **False Positive Rate** | 1.71% | **65.99%** | **65.99%** | 50.51% |
| **ROC-AUC** | 1.0000 | **0.5310** | **0.5310** | **0.6728** |
| **PR-AUC (Average Precision)** | 1.0000 | **0.0734** | — | **0.0981** |
| **Confusion Matrix** | TN=288, FP=5, FN=0, TP=293 | **TN=101, FP=196, FN=5, TP=12** | Projected: TP=73, FP=967 | TN=147, FP=150, FN=6, TP=11 |
| **Release Verdict** | PASSED *(fraudulent)* | **REJECTED (NO-GO) ❌** | **REJECTED (NO-GO) ❌** | Shadow-mode Candidate |

---

## 🚀 Quickstart & Reproduction

### Prerequisites
- Python 3.10+
- PyTorch 2.0+
- scikit-learn, imbalanced-learn, pandas, numpy, joblib

### Running the Notebook
```bash
git clone https://github.com/shivam-more1609/GDG_WildCard.git
cd wafer-guard-fixed
jupyter notebook wafer_guard_fixed.ipynb
```
The notebook is self-contained and pre-executed with all cell outputs saved.
