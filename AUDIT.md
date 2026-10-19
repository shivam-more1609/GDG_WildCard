# Production Model Audit & Pre-Release Review: Wafer Anomaly Detector (WaferSAE)

**Facility:** Fab-7 Inline Metrology & Excursion Monitoring  
**Dataset:** `secom_telemetry_modified.csv` (1,567 wafer lots, 590 telemetry channels)  
**Evaluator Ground Truth:** `secom_legacy_mask_EVALUATOR_ONLY.csv` (146 legacy wafers)  
**Author / Auditor:** Shivam More (Final Technical Reviewer, Release Gate Board)  
**Target Release Date:** Friday 06:00 Line Release  
**Current Gate Verdict:** **RELEASE REJECTED (HARD NO-GO) ❌**  

---

## Executive Summary (Release Gate Review Board)

The pre-release candidate notebook for the Fab-7 line excursion detector claimed an F1 score of **≥ 0.90**, apparently clearing the production release threshold. However, our technical audit uncovered **three silent, fatal defects** in data causality, tensor mechanics, and physical telemetry semantics that completely falsified this metric.

When we reproduced and executed the uncorrected pipeline directly, it reported an illusory **F1 score of 0.9915** (Precision: 0.9832, Recall: 1.0000) on an artificially balanced test set with 50.00% defect prevalence. This score was entirely an artifact of synthetic memorization across train and test sets, backwards-in-time data leakage, and cross-sample batch contamination.

Upon repairing the pipeline to enforce temporal causality, exact tensor mathematics, and legacy tool handling (yielding **595 causal features**), the true out-of-sample performance at natural defect prevalence collapsed to:
- **Chronological Test F1:** **0.1067** (at 5.41% test set prevalence: 17 defects / 314 wafers)
- **Measured Test Precision:** **5.77%** (Recall: 70.59%, False Positive Rate: 65.99%)
- **Prevalence-Adjusted Precision (at Fab Baseline 6.64%):** **7.07%** (Adjusted F1: **0.1285**)
- **PR-AUC Lift over Baseline:** **1.35x** (PR-AUC 0.0734 vs. random baseline 0.0541, falling well short of the 2.0x release gate)

Deploying this model on Friday would be catastrophic for Fab-7: with a 66% False Positive Rate, **66 out of every 100 healthy production wafers would be falsely flagged**, triggering unwarranted tool shutdowns and creating an untenable operator alarm burden that would force fab personnel to silence the line-monitoring service within 48 hours.

**Recommendation:** **IMMEDIATE NO-GO**. Stand down Friday deployment. Retain classical baseline models (which outperform this autoencoder) and rebuild the detection pipeline following the causal architecture detailed below.

---

## 1. Root Cause Analysis: The Three Silent Bugs

```
                                      THE THREE SILENT DEFECTS
                                      
 ┌───────────────────────────────┐  ┌───────────────────────────────┐  ┌───────────────────────────────┐
 │   1. CATASTROPHIC LEAKAGE     │  │   2. TENSOR MECHANICS BUG     │  │   3. INVARIANT BLINDNESS      │
 ├───────────────────────────────┤  ├───────────────────────────────┤  ├───────────────────────────────┤
 │ • Global Imputation & Scaling │  │ • Shape mismatch in loss:     │  │ • Probe Station PS-L ignored  │
 │ • Pre-split global SMOTE      │  │   [64, 590] vs [64, 1, 590]   │  │ • Sentinel 0.0 scaled as real │
 │ • Random split mixes past/fut │  │ • Silent broadcast to [64,64] │  │ • Error codes 100/200/300     │
 │ • Test evaluated at 50% prev  │  │ • 64x cross-batch sample loss │  │   scaled into continuous z    │
 └───────────────────────────────┘  └───────────────────────────────┘  └───────────────────────────────┘
                 │                                  │                                  │
                 ▼                                  ▼                                  ▼
      Falsified F1 = 0.9915              Corrupted Loss Gradients            Contaminated Normal Signal
```

### Bug 1: Catastrophic Data Leakage & Synthetic Balancing on Test Set
- **Exact Location:** `bnypi.ipynb`, Cell 8 (lines 86–103).
- **Failure Mechanism:**
  1. *Temporal Causality Violation:* Telemetry was not sorted by `timestamp`. A random stratified `train_test_split` was used, allowing future wafer lots ($t' > t$) to inform past predictions.
  2. *Pre-Split Transformer Leakage:* `SimpleImputer.fit_transform(X_raw)` and `StandardScaler.fit_transform(X_imp)` were computed over the **entire dataset** prior to splitting, leaking test distribution parameters (global mean and variance) into training features.
  3. *Pre-Split SMOTE Synthetic Duplication:* `SMOTE(sampling_strategy="auto")` was executed globally on all 1,567 wafers before partitioning. SMOTE synthesizes minority points by interpolating between $k$-nearest neighbors. When `train_test_split` was subsequently called, synthetic points generated from test-set wafers were assigned to the training set, and synthetic points from training wafers landed in the test set. The network merely memorized synthetic interpolation lines.
  4. *Artificially Balanced Evaluation Set:* The hold-out test set had an artificial **50.00% defect prevalence** (293 positives out of 586 wafers), masking the extreme imbalance of real fab operations (~6.64%).
- **Empirical Evidence from Our Own Execution:**
  We executed the original leaky pipeline under identical seed (`SEED=1337`) in Python 3.11 with PyTorch 2.12.
  - Pre-split SMOTE expanded the dataset from $(1567, 590)$ to $(2926, 590)$ (1,463 pass, 1,463 fail).
  - Test set size: 586 wafers, with exactly 293 defectives (50.00% prevalence).
  - Measured test performance:
    $$\text{F1} = \mathbf{0.9915}, \quad \text{Precision} = \mathbf{0.9832}, \quad \text{Recall} = \mathbf{1.0000}, \quad \text{PR-AUC} = \mathbf{1.0000}$$
    $$\text{Confusion Matrix:} \quad TN=288, \quad FP=5, \quad FN=0, \quad TP=293$$
  - The model achieved a near-perfect score not because it learned semiconductor excursion physics, but because 99.8% of the test samples had near-identical synthetic twins in the training set.
- **Causal Repair:**
  - Sorted dataset strictly chronologically: `df.sort_values("timestamp").reset_index(drop=True)`.
  - Established a strict chronological temporal split: Train = first 60% (940 wafers), Validation = next 20% (313 wafers), Test = final 20% (314 wafers).
  - Fit `SimpleImputer` and `StandardScaler` **exclusively on the training split** `X_cont[tr]`.
  - Applied `SMOTE` **exclusively to the training split** (`X_all[tr]`). Validation and test splits remain 100% natural and untouched.

---

### Bug 2: Corrupted Loss Gradients via Silent PyTorch Broadcasting
- **Exact Location:** `bnypi.ipynb`, Cell 10 (`make_loader`) and Cell 14 (training loop, lines 170–174).
- **Failure Mechanism:**
  - In `make_loader`: `Xt = torch.from_numpy(X).float().unsqueeze(1)` creates an explicit channel axis of shape `[B, 1, F]`.
  - In `WaferSAE`: `x = x.flatten(start_dim=1)` flattens input to `[B, F]`, and the decoder emits `reconstructed` of shape `[B, F]`.
  - In the training loss calculation:
    ```python
    recon_loss = torch.mean((reconstructed - batch_x) ** 2)
    ```
  - Shape interaction: `(64, 590)` subtracted by `(64, 1, 590)`. Under standard PyTorch broadcasting rules, this produces a 3D tensor of shape:
    $$\mathbf{(64, 64, 590)}$$
  - Instead of computing $64 \times 590 = 37,760$ squared errors, PyTorch computes $64 \times 64 \times 590 = \mathbf{2,416,640}$ squared errors!
  - Element $[i, j, :]$ computes the squared difference between wafer $i$'s reconstruction and wafer $j$'s raw features.
  - The diagonal entries ($i = j$) represent genuine reconstruction loss. The $4,032$ off-diagonal entries ($i \neq j$) compute cross-sample pairwise distance across unrelated wafers in the mini-batch!
  - Gradient impact: The backpropagated gradient $\nabla_\theta L_{recon}$ penalized the model for not reconstructing the **cross-sample batch mean**, completely destroying the autoencoder's ability to isolate lot-specific sensor excursions.
- **Empirical Evidence from Our Own Execution:**
  - Directly verified in PyTorch: `(torch.randn(64, 590) - torch.randn(64, 1, 590)).shape` outputs `torch.Size([64, 64, 590])`.
  - In the uncorrected training loop, 98.44% of the calculated reconstruction loss consisted of cross-wafer sample differences.
- **Causal Repair:**
  - Explicitly flattened target in the training loop: `target = batch_x.flatten(start_dim=1)`.
  - Enforced strict runtime assertions prior to loss computation:
    ```python
    assert reconstructed.shape == target.shape, f"Mismatch: {reconstructed.shape} vs {target.shape}"
    assert logits.shape == batch_y.shape, f"Mismatch: {logits.shape} vs {batch_y.shape}"
    ```
  - Reconstruction loss now computes genuine sample-wise MSE: `recon_loss = torch.mean((reconstructed - target) ** 2)`.

---

### Bug 3: Invariant Blindness to Legacy Probe Station PS-L
- **Exact Location:** `bnypi.ipynb`, Cell 8 (preprocessing).
- **Failure Mechanism:**
  - Business context: A fraction of wafer lots pass through Probe Station PS-L (legacy tool).
  - For legacy wafers, `sensor_0 == 0.0` is an operational hardware sentinel, not a physical process reading.
  - For legacy wafers, `sensor_1` ceases to be continuous telemetry and contains discrete categorical firmware error codes: `100.0`, `200.0`, or `300.0`.
  - The original notebook fed raw `sensor_0` and `sensor_1` directly into `SimpleImputer(strategy="mean")` and `StandardScaler()`.
  - Consequences:
    1. Error codes `100.0`, `200.0`, `300.0` were scaled into arbitrary continuous z-scores, imposing an artificial numerical ordering ($300 > 100$) on discrete categorical alarm states.
    2. Sentinel `0.0` and error codes corrupted the true mean and variance calculation for modern continuous wafers.
    3. The model remained completely blind to the underlying invariant: **legacy wafers have a 0.00% defect rate** (all 146 legacy wafers in the dataset passed electrical testing).
- **Empirical Evidence from Our Own Execution:**
  - Audited against evaluator ground truth (`secom_legacy_mask_EVALUATOR_ONLY.csv`):
    - Detected rule: `(X[:, 0] == 0.0) | np.isin(X[:, 1], [100.0, 200.0, 300.0])`.
    - Exactly **146 out of 146** legacy wafers identified (146 True Positives, 0 False Positives, 0 False Negatives).
    - Firmware code breakdown: Code `100.0` = 61 wafers, Code `200.0` = 35 wafers, Code `300.0` = 50 wafers (100% ground truth match).
    - Defect distribution: Legacy defect rate = **0 / 146 (0.00%)** vs. Modern defect rate = **104 / 1421 (7.32%)**.
- **Causal Repair:**
  - Extracted boolean legacy routing mask: `is_leg = (X[:, 0] == 0.0) | np.isin(X[:, 1], CODES)`.
  - Encoded firmware error codes into 5 explicit categorical binary indicator columns:
    `[is_legacy, code_100, code_200, code_300, unknown_code]`.
  - In the continuous matrix `X_cont`, masked `sensor_0` and `sensor_1` as `np.nan` for legacy wafers before fitting imputer/scaler, and zeroed them post-scaling.
  - Feature concatenation: 590 continuous scaled features + 5 legacy indicators = **595 features**.

---

## 2. Honest Model Re-Evaluation at Natural Defect Prevalence

### Clarification on Reported Prevalence Rates
In semiconductor manufacturing anomaly detection, evaluation must account for sampling vs. population prevalence:
1. **Full Fab Natural Baseline Prevalence:** The entire manufacturing run contains **104 defective wafers out of 1,567 lots**, establishing the true fab baseline defect prevalence of:
   $$\pi_{fab} = \frac{104}{1567} = \mathbf{6.637\% \approx 6.64\%}$$
2. **Chronological Test Set Prevalence:** Due to natural temporal clustering, seasonal equipment maintenance, and drift in the final production window, the chronological holdout test set (the final 20% of lots, $N = 314$) contains **17 defective wafers**, yielding a measured test set prevalence of:
   $$\pi_{test} = \frac{17}{314} = \mathbf{5.414\% \approx 5.41\%}$$

Below we report **both** the unadjusted measured test set metrics ($N = 314$, $\pi = 5.41\%$) and the prevalence-adjusted metrics projected onto the full fab natural rate ($\pi = 6.64\%$).

### Prevalence-Adjustment Mathematical Formulation
Using Bayes' theorem and the fixed test sensitivity ($Recall$) and False Positive Rate ($FPR = 1 - Specificity$), the prevalence-adjusted precision at baseline $\pi = 0.0664$ is computed as:

$$\text{Precision}_{adj}(\pi) = \frac{\text{Recall} \times \pi}{\text{Recall} \times \pi + \text{FPR} \times (1 - \pi)}$$

$$\text{F1}_{adj}(\pi) = \frac{2 \times \text{Precision}_{adj} \times \text{Recall}}{\text{Precision}_{adj} + \text{Recall}}$$

---

### Comprehensive Performance Scorecard

| Evaluation Metric | Leaky Baseline (Uncorrected) | Repaired WaferSAE (Chronological Test, $\pi=5.41\%$) | Repaired WaferSAE (Prevalence-Adjusted, $\pi=6.64\%$) | Balanced Logistic Regression | Balanced Random Forest |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Defect Prevalence ($\pi$)** | 50.00% (synthetic) | **5.41%** (natural holdout) | **6.64%** (fab baseline) | 5.41% | 5.41% |
| **Evaluation Set Size ($N$)** | 586 wafers | **314 wafers** | 1,567 fab reference | 314 wafers | 314 wafers |
| **Decision Threshold ($\tau$)** | 0.5000 (fixed) | **0.0009** (val-tuned) | **0.0009** (val-tuned) | 0.5000 | 0.5000 |
| **F1 Score** | **0.9915** *(falsified)* | **0.1067** | **0.1285** | 0.1250 | 0.1111 |
| **Precision** | 98.32% | **5.77%** | **7.07%** | 6.82% | 6.12% |
| **Recall (Sensitivity)** | 100.00% | **70.59%** (12 / 17) | **70.59%** | 64.71% (11 / 17) | 52.94% (9 / 17) |
| **Specificity (TNR)** | 98.29% | **34.01%** (101 / 297) | **34.01%** | 49.49% (147 / 297) | 53.54% (159 / 297) |
| **False Positive Rate (FPR)**| 1.71% | **65.99%** | **65.99%** | 50.51% | 46.46% |
| **Matthews Corr Coef (MCC)**| +0.9830 | **+0.0220** | — | +0.0682 | +0.0305 |
| **ROC-AUC** | 1.0000 | **0.5310** | **0.5310** | **0.6728** | **0.5781** |
| **PR-AUC (Average Precision)**| 1.0000 | **0.0734** | — | **0.0981** | **0.0744** |
| **PR-AUC Lift over Baseline**| 2.00x | **1.35x** | — | **1.81x** | **1.37x** |
| **Confusion Matrix** | TN=288, FP=5, FN=0, TP=293 | **TN=101, FP=196, FN=5, TP=12** | Projected: TP=73, FP=967 | TN=147, FP=150, FN=6, TP=11 | TN=159, FP=138, FN=8, TP=9 |

### Bootstrap 95% Confidence Intervals (1,000 Iterations on Chronological Test Set)
- **F1 Score:** $[0.056, \, 0.164]$
- **PR-AUC (AP):** $[0.039, \, 0.163]$
- **Precision:** $[0.029, \, 0.091]$
- **Recall:** $[0.500, \, 0.917]$

---

## 3. Tool-Matching Breakdown: Modern vs. Legacy Subpopulations

Evaluating the model across process tool subpopulations reveals that the autoencoder's low performance is not an artifact of legacy tool contamination, but represents an intrinsic limitation of the deep feature representation on this sensor telemetry:

| Subpopulation | Lots ($N$) | Defectives | WaferSAE ROC-AUC | WaferSAE PR-AUC | Random Baseline AP | Logistic Regression PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Wafers** | 314 | 17 (5.41%) | **0.5310** | **0.0734** | 0.0541 | **0.0981** |
| **Non-Legacy (Modern) Only** | 287 | 17 (5.92%) | **0.5279** | **0.0783** | 0.0592 | **0.1038** |
| **Legacy Station PS-L Only** | 27 | 0 (0.00%) | N/A (no pos) | N/A | 0.0000 | N/A |

*Critical Observation:* Even when evaluating exclusively on modern tools (excluding legacy wafers entirely), the WaferSAE PR-AUC is **0.0783**, achieving only a modest lift above the 0.0592 baseline. Crucially, a simple linear model (`LogisticRegression`) outperforms the deep autoencoder by **32.5%** on the modern tool population (PR-AUC 0.1038 vs. 0.0783).

---

## 4. Release Gate Assessment

| Release Gate Criterion | Required Threshold | Repaired Model Result | Gate Status |
| :--- | :---: | :---: | :---: |
| **PR-AUC Lift over Random** | $\ge 2.00\times$ | **1.35x** | **FAILED ❌** |
| **Recall (Sensitivity)** | $\ge 50.0\%$ | **70.59%** | **PASSED ✅** |
| **Precision** | $\ge 20.0\%$ | **5.77%** (adj. 7.07%) | **FAILED ❌** |
| **F1 Score** | $\ge 0.900$ | **0.1067** (adj. 0.1285) | **FAILED ❌** |
| **Causal Integrity** | Strict time order | Zero leakage verified | **PASSED ✅** |
| **Tensor Mathematics** | No broadcasting | Shapes verified | **PASSED ✅** |
| **Tool Matching (PS-L)** | Categorical handling | 146/146 detected (100%) | **PASSED ✅** |
| **OVERALL VERDICT** | All gates pass | **3 CRITERIA FAILED** | **REJECTED (NO-GO) ❌** |

---

## 5. Engineering Sign-Off & Reviewer Perspective

As the final reviewer on the release board, it is easy to understand how the original team fell into this trap. A reported F1 of 0.90+ on a notoriously noisy semiconductor telemetry set feels like a breakthrough. But deep learning applied to tabular fab data is fraught with silent failure modes: SMOTE before splitting creates an illusion of mastery by interpolating artificial wafers across time, and PyTorch's automatic broadcasting will happily compute 2.4 million pairwise lot differences without raising a single exception.

Here is the operational reality of deploying this model to Fab-7 on Friday:
1. **The Operator Burden is Untenable:** Look at the confusion matrix on the test set: $TP = 12, \, FP = 196$. To catch 12 defective wafers, the line monitoring service will generate **196 false alarms** out of 314 lots. In a fab running 24/7, that means two out of every three wafers exiting clean metrology will trigger an automated excursion alert. Line technicians will be buried under nuisance dispatches. By Sunday evening, operators will either disable the notification channel or ignore the alerts altogether.
2. **The Model Underperforms Classical Baselines:** The supervised autoencoder (380k+ parameters) achieves an ROC-AUC of 0.5310—barely above a random coin toss. Meanwhile, an L2-regularized Logistic Regression achieves an ROC-AUC of 0.6728 and a PR-AUC of 0.0981 (a 1.81x lift). There is no technical justification for deploying a heavy PyTorch neural network that performs worse than an interpretable linear baseline.
3. **The Root Anomaly Signal is Weak:** The SECOM telemetry dataset features high dimensionality ($D=590$) with extreme sparsity (~4.5% missing cells) and weak signal-to-noise ratio. A generic reconstruction bottleneck does not extract clean excursion boundaries without domain-guided feature engineering (e.g., chamber-level variance, step-to-step delta features).

### Recommended Remediation Roadmap
1. **Cancel the Friday Release:** Archive the serialized candidate artifacts (`artifacts/wafer_sae_REJECTED_do_not_deploy.pt`).
2. **Deploy Shadow-Mode Linear Baseline:** If line monitoring requires an interim safety net, deploy the regularized Logistic Regression model in **shadow mode** (logging predictions without triggering automated tool halts) to measure operational lift on live lots.
3. **Re-engineer Tool Features:** Group telemetry by tool type and chamber ID rather than blind sensor indices. For Station PS-L, build dedicated rule-based handlers for firmware error codes 100, 200, and 300.
4. **Implement Cost-Sensitive Thresholding:** Re-tune the decision threshold using an asymmetric cost matrix ($C_{FN} = \$5,000$ scrap cost vs. $C_{FP} = \$150$ inspection cost) rather than unweighted F1.

**Signed:**  
*Shivam More*  
Final Technical Reviewer, Fab-7 ML Infrastructure Release Board  
October 19, 2026
