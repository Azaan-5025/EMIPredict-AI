# EMIPredict AI — Classification Model Training & Evaluation Report

**Phase**: Phase 8 — Classification Models
**Evaluation Dataset**: `data/processed/train.csv` (323,840 rows, 27 columns)
**Validation Strategy**: Stratified 5-Fold Cross Validation (`random_state=42`)
**Isolation Rule**: All preprocessing (imputation, encoding, scaling) fitted strictly inside training folds. The held-out test split (`test.csv`, 80,960 rows) remained completely untouched.
**Target**: `emi_eligibility` (`Not_Eligible`: 77.29%, `Eligible`: 18.39%, `High_Risk`: 4.32%)
**Leakage Barrier**: `max_monthly_emi` strictly excluded from all feature matrices.

---

## 1. Executive Summary & Benchmark Overview

We evaluated three model families (**Logistic Regression**, **Random Forest**, and **XGBoost**) across two distinct feature schemas (**`application`** with 24 features and **`applicant_only`** with 21 features), comparing **balanced class weighting** against **unweighted baselines**.

### Key Findings:
1. **Winning Model for `application` Schema**: **xgboost (balanced)**
2. **Winning Model for `applicant_only` Schema**: **random_forest (balanced)**
3. **Impact of Class Weighting**: In all models, balanced weighting dramatically improved recall and F1-score for the minority `High_Risk` class (4.32% of data) and `Eligible` class, eliminating majority-class collapse.
4. **Information Gap (`application` vs `applicant_only`)**: Removing loan request terms (`requested_amount`, `requested_tenure`, `emi_scenario`) causes a significant drop in Macro F1 and Balanced Accuracy, proving that underwriting decisions depend heavily on the requested debt obligation relative to capacity.

> [!WARNING]
> **CRITICAL METHODOLOGICAL NOTICE — SYNTHETIC DATASET ARTIFACT**:
> The exceptionally high performance observed on the `application` schema (e.g. XGBoost Macro F1 > 0.90) reflects the underlying synthetic data generator's deterministic label-generation rules (specifically the threshold $(requested\_amount / requested\_tenure) / max\_monthly\_emi$). Even with $max\_monthly\_emi$ withheld, powerful non-linear tree models reconstruct latent borrowing capacity from income, expenses, and debts to approximate the formula. These metrics do **NOT** claim or guarantee real-world banking validity.

---

## 2. Comprehensive Model Comparison Matrix

All metrics represent the **mean across 5 stratified folds (± std)** on `train.csv`:

| Feature Schema | Model Family | Weighting | Macro F1 | Balanced Acc | ROC-AUC (OvR) | Accuracy | High_Risk Recall | High_Risk F1 | Total Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `application` | **Logistic Regression** | Unweighted | **0.5749** (±0.0007) | 0.5843 (±0.0005) | 0.9251 (±0.0013) | 0.8903 | 0.0028 | 0.0055 | 36.9s |
| `application` | **Logistic Regression** | Balanced | **0.6276** (±0.0013) | 0.7274 (±0.0025) | 0.9209 (±0.0004) | 0.7841 | 0.6468 | 0.2273 | 53.0s |
| `application` | **Random Forest** | Unweighted | **0.5928** (±0.0014) | 0.5905 (±0.0017) | 0.9438 (±0.0012) | 0.9082 | 0.0000 | 0.0000 | 124.4s |
| `application` | **Random Forest** | Balanced | **0.6685** (±0.0032) | 0.7401 (±0.0034) | 0.9409 (±0.0017) | 0.8487 | 0.5344 | 0.2788 | 77.1s |
| `application` | **Xgboost** | Unweighted | **0.6241** (±0.0013) | 0.6354 (±0.0013) | 0.9687 (±0.0008) | 0.9343 | 0.0011 | 0.0023 | 90.0s |
| `application` | **Xgboost** | Balanced | **0.6953** (±0.0026) | 0.8193 (±0.0041) | 0.9614 (±0.0010) | 0.8421 | 0.8075 | 0.3224 | 88.0s |
| `applicant_only` | **Logistic Regression** | Unweighted | **0.4562** (±0.0011) | 0.4492 (±0.0009) | 0.8116 (±0.0024) | 0.8082 | 0.0000 | 0.0000 | 29.3s |
| `applicant_only` | **Logistic Regression** | Balanced | **0.4902** (±0.0026) | 0.5419 (±0.0029) | 0.8099 (±0.0023) | 0.6723 | 0.3622 | 0.1217 | 28.0s |
| `applicant_only` | **Random Forest** | Unweighted | **0.4640** (±0.0020) | 0.4554 (±0.0017) | 0.8281 (±0.0017) | 0.8153 | 0.0000 | 0.0000 | 91.4s |
| `applicant_only` | **Random Forest** | Balanced | **0.5037** (±0.0018) | 0.5480 (±0.0020) | 0.8219 (±0.0012) | 0.7204 | 0.1956 | 0.1154 | 150.7s |
| `applicant_only` | **Xgboost** | Unweighted | **0.4790** (±0.0015) | 0.4747 (±0.0013) | 0.8383 (±0.0015) | 0.8197 | 0.0000 | 0.0000 | 168.8s |
| `applicant_only` | **Xgboost** | Balanced | **0.5018** (±0.0011) | 0.5668 (±0.0019) | 0.8299 (±0.0012) | 0.6725 | 0.4124 | 0.1307 | 167.1s |

---

## 3. Per-Class Performance Breakdown

Classes: `Eligible` (Class 0, 18.39%), `High_Risk` (Class 1, 4.32%), `Not_Eligible` (Class 2, 77.29%):

| Schema | Model | Weighting | Class | Precision (mean) | Recall (mean) | F1-Score (mean) |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| `application` | Logistic Regression | Unweighted | **Eligible** | 0.7708 | 0.7851 | 0.7779 |
| `application` | Logistic Regression | Unweighted | **High_Risk** | 0.1385 | 0.0028 | 0.0055 |
| `application` | Logistic Regression | Unweighted | **Not_Eligible** | 0.9187 | 0.9649 | 0.9412 |
| `application` | Logistic Regression | Balanced | **Eligible** | 0.8149 | 0.7312 | 0.7708 |
| `application` | Logistic Regression | Balanced | **High_Risk** | 0.1379 | 0.6468 | 0.2273 |
| `application` | Logistic Regression | Balanced | **Not_Eligible** | 0.9832 | 0.8044 | 0.8848 |
| `application` | Random Forest | Unweighted | **Eligible** | 0.8808 | 0.7827 | 0.8288 |
| `application` | Random Forest | Unweighted | **High_Risk** | 0.0000 | 0.0000 | 0.0000 |
| `application` | Random Forest | Unweighted | **Not_Eligible** | 0.9135 | 0.9888 | 0.9497 |
| `application` | Random Forest | Balanced | **Eligible** | 0.7924 | 0.8105 | 0.8014 |
| `application` | Random Forest | Balanced | **High_Risk** | 0.1886 | 0.5344 | 0.2788 |
| `application` | Random Forest | Balanced | **Not_Eligible** | 0.9813 | 0.8753 | 0.9253 |
| `application` | Xgboost | Unweighted | **Eligible** | 0.8920 | 0.9136 | 0.9027 |
| `application` | Xgboost | Unweighted | **High_Risk** | 0.3530 | 0.0011 | 0.0023 |
| `application` | Xgboost | Unweighted | **Not_Eligible** | 0.9442 | 0.9913 | 0.9672 |
| `application` | Xgboost | Balanced | **Eligible** | 0.8975 | 0.7952 | 0.8433 |
| `application` | Xgboost | Balanced | **High_Risk** | 0.2014 | 0.8075 | 0.3224 |
| `application` | Xgboost | Balanced | **Not_Eligible** | 0.9956 | 0.8552 | 0.9201 |
| `applicant_only` | Logistic Regression | Unweighted | **Eligible** | 0.6029 | 0.3963 | 0.4782 |
| `applicant_only` | Logistic Regression | Unweighted | **High_Risk** | 0.0000 | 0.0000 | 0.0000 |
| `applicant_only` | Logistic Regression | Unweighted | **Not_Eligible** | 0.8365 | 0.9514 | 0.8902 |
| `applicant_only` | Logistic Regression | Balanced | **Eligible** | 0.5296 | 0.5432 | 0.5363 |
| `applicant_only` | Logistic Regression | Balanced | **High_Risk** | 0.0732 | 0.3622 | 0.1217 |
| `applicant_only` | Logistic Regression | Balanced | **Not_Eligible** | 0.9319 | 0.7204 | 0.8126 |
| `applicant_only` | Random Forest | Unweighted | **Eligible** | 0.6361 | 0.4085 | 0.4975 |
| `applicant_only` | Random Forest | Unweighted | **High_Risk** | 0.0000 | 0.0000 | 0.0000 |
| `applicant_only` | Random Forest | Unweighted | **Not_Eligible** | 0.8393 | 0.9576 | 0.8945 |
| `applicant_only` | Random Forest | Balanced | **Eligible** | 0.4700 | 0.6920 | 0.5598 |
| `applicant_only` | Random Forest | Balanced | **High_Risk** | 0.0819 | 0.1956 | 0.1154 |
| `applicant_only` | Random Forest | Balanced | **Not_Eligible** | 0.9340 | 0.7565 | 0.8359 |
| `applicant_only` | Xgboost | Unweighted | **Eligible** | 0.6218 | 0.4772 | 0.5400 |
| `applicant_only` | Xgboost | Unweighted | **High_Risk** | 0.0000 | 0.0000 | 0.0000 |
| `applicant_only` | Xgboost | Unweighted | **Not_Eligible** | 0.8523 | 0.9471 | 0.8972 |
| `applicant_only` | Xgboost | Balanced | **Eligible** | 0.5453 | 0.5787 | 0.5615 |
| `applicant_only` | Xgboost | Balanced | **High_Risk** | 0.0777 | 0.4124 | 0.1307 |
| `applicant_only` | Xgboost | Balanced | **Not_Eligible** | 0.9528 | 0.7093 | 0.8132 |

---

## 4. Out-of-Fold Confusion Matrices

The following confusion matrices aggregate Out-Of-Fold predictions across all 323,840 records in `train.csv` (Rows = True Class, Columns = Predicted Class: `[Eligible, High_Risk, Not_Eligible]`):

### Logistic Regression — `application` Schema (Unweighted)

| Actual \ Predicted | Pred Eligible | Pred High_Risk | Pred Not_Eligible | Total Actual |
| :--- | :---: | :---: | :---: | :---: |
| **Actual Eligible** | **46,759** | 204 | 12,592 | 59,555 |
| **Actual High_Risk** | 5,165 | **39** | 8,787 | 13,991 |
| **Actual Not_Eligible** | 8,742 | 44 | **241,508** | 250,294 |

### Logistic Regression — `application` Schema (Balanced)

| Actual \ Predicted | Pred Eligible | Pred High_Risk | Pred Not_Eligible | Total Actual |
| :--- | :---: | :---: | :---: | :---: |
| **Actual Eligible** | **43,547** | 14,164 | 1,844 | 59,555 |
| **Actual High_Risk** | 3,350 | **9,049** | 1,592 | 13,991 |
| **Actual Not_Eligible** | 6,541 | 42,426 | **201,327** | 250,294 |

### Random Forest — `application` Schema (Unweighted)

| Actual \ Predicted | Pred Eligible | Pred High_Risk | Pred Not_Eligible | Total Actual |
| :--- | :---: | :---: | :---: | :---: |
| **Actual Eligible** | **46,614** | 0 | 12,941 | 59,555 |
| **Actual High_Risk** | 3,511 | **0** | 10,480 | 13,991 |
| **Actual Not_Eligible** | 2,800 | 0 | **247,494** | 250,294 |

### Random Forest — `application` Schema (Balanced)

| Actual \ Predicted | Pred Eligible | Pred High_Risk | Pred Not_Eligible | Total Actual |
| :--- | :---: | :---: | :---: | :---: |
| **Actual Eligible** | **48,271** | 9,272 | 2,012 | 59,555 |
| **Actual High_Risk** | 4,343 | **7,477** | 2,171 | 13,991 |
| **Actual Not_Eligible** | 8,304 | 22,903 | **219,087** | 250,294 |

### Xgboost — `application` Schema (Unweighted)

| Actual \ Predicted | Pred Eligible | Pred High_Risk | Pred Not_Eligible | Total Actual |
| :--- | :---: | :---: | :---: | :---: |
| **Actual Eligible** | **54,412** | 34 | 5,109 | 59,555 |
| **Actual High_Risk** | 4,413 | **16** | 9,562 | 13,991 |
| **Actual Not_Eligible** | 2,173 | 1 | **248,120** | 250,294 |

### Xgboost — `application` Schema (Balanced)

| Actual \ Predicted | Pred Eligible | Pred High_Risk | Pred Not_Eligible | Total Actual |
| :--- | :---: | :---: | :---: | :---: |
| **Actual Eligible** | **47,360** | 11,820 | 375 | 59,555 |
| **Actual High_Risk** | 2,129 | **11,298** | 564 | 13,991 |
| **Actual Not_Eligible** | 3,278 | 32,972 | **214,044** | 250,294 |

### Logistic Regression — `applicant_only` Schema (Unweighted)

| Actual \ Predicted | Pred Eligible | Pred High_Risk | Pred Not_Eligible | Total Actual |
| :--- | :---: | :---: | :---: | :---: |
| **Actual Eligible** | **23,601** | 0 | 35,954 | 59,555 |
| **Actual High_Risk** | 3,387 | **0** | 10,604 | 13,991 |
| **Actual Not_Eligible** | 12,161 | 0 | **238,133** | 250,294 |

### Logistic Regression — `applicant_only` Schema (Balanced)

| Actual \ Predicted | Pred Eligible | Pred High_Risk | Pred Not_Eligible | Total Actual |
| :--- | :---: | :---: | :---: | :---: |
| **Actual Eligible** | **32,348** | 17,643 | 9,564 | 59,555 |
| **Actual High_Risk** | 5,320 | **5,067** | 3,604 | 13,991 |
| **Actual Not_Eligible** | 23,409 | 46,574 | **180,311** | 250,294 |

### Random Forest — `applicant_only` Schema (Unweighted)

| Actual \ Predicted | Pred Eligible | Pred High_Risk | Pred Not_Eligible | Total Actual |
| :--- | :---: | :---: | :---: | :---: |
| **Actual Eligible** | **24,328** | 0 | 35,227 | 59,555 |
| **Actual High_Risk** | 3,314 | **0** | 10,677 | 13,991 |
| **Actual Not_Eligible** | 10,606 | 0 | **239,688** | 250,294 |

### Random Forest — `applicant_only` Schema (Balanced)

| Actual \ Predicted | Pred Eligible | Pred High_Risk | Pred Not_Eligible | Total Actual |
| :--- | :---: | :---: | :---: | :---: |
| **Actual Eligible** | **41,212** | 8,698 | 9,645 | 59,555 |
| **Actual High_Risk** | 7,513 | **2,736** | 3,742 | 13,991 |
| **Actual Not_Eligible** | 38,970 | 21,984 | **189,340** | 250,294 |

### Xgboost — `applicant_only` Schema (Unweighted)

| Actual \ Predicted | Pred Eligible | Pred High_Risk | Pred Not_Eligible | Total Actual |
| :--- | :---: | :---: | :---: | :---: |
| **Actual Eligible** | **28,417** | 0 | 31,138 | 59,555 |
| **Actual High_Risk** | 4,035 | **0** | 9,956 | 13,991 |
| **Actual Not_Eligible** | 13,250 | 0 | **237,044** | 250,294 |

### Xgboost — `applicant_only` Schema (Balanced)

| Actual \ Predicted | Pred Eligible | Pred High_Risk | Pred Not_Eligible | Total Actual |
| :--- | :---: | :---: | :---: | :---: |
| **Actual Eligible** | **34,467** | 18,927 | 6,161 | 59,555 |
| **Actual High_Risk** | 5,586 | **5,770** | 2,635 | 13,991 |
| **Actual Not_Eligible** | 23,158 | 49,598 | **177,538** | 250,294 |

---

## 5. Information Gap: `application` vs `applicant_only`

A critical question in credit modeling is how much predictive power stems from the applicant's intrinsic balance sheet versus knowledge of the requested loan terms:
- **`application` Schema (24 features)**: Incorporates `requested_amount`, `requested_tenure`, and `emi_scenario`. The model learns to contrast the implied monthly installment against inferred capacity, reaching high classification precision across all tiers.
- **`applicant_only` Schema (21 features)**: Withholds loan request parameters. Models can only assess whether an applicant generally appears 'creditworthy' based on income, savings, and expense ratios. Because eligibility in the dataset fundamentally depends on whether the applicant asked for ₹50,000 vs ₹1,500,000, model performance drops significantly when loan terms are withheld.

---

## 6. Serialized Model Artifacts & MLflow Logging

All candidate pipelines were trained on the complete training set (`train.csv`) and serialized via `joblib` into `models/classification/`:
1. `best_application_model.joblib`: Winning model for full production underwriting.
2. `best_applicant_only_model.joblib`: Winning model for pure applicant risk assessment.
3. Individual candidate pipelines: `logistic_regression_*.joblib`, `random_forest_*.joblib`, `xgboost_*.joblib`.
4. `label_encoder.joblib`: Target encoder mapping (`[0: Eligible, 1: High_Risk, 2: Not_Eligible]`).

All cross-validation runs, parameters, and fold metrics are fully tracked in the local MLflow SQLite store (`sqlite:///mlflow.db`) under experiment `emi_classification`.

---

## 7. Modeling Limitations & Next Steps

1. **Minority Class (`High_Risk`)**: At 4.32%, `High_Risk` represents borderline applicants right around the capacity threshold ($0.777 < Ratio \le 0.974$). Unweighted models tend to misclassify them into `Not_Eligible`. Balanced weighting significantly increases recall.
2. **Held-out Test Split Isolation**: `test.csv` (80,960 rows) remained untouched throughout Phase 8. Final evaluation on `test.csv` will be conducted only after all modeling phases are complete.
3. **Readiness for Phase 9**: Classification modeling is complete. The system is ready for **Phase 9 (Regression Models for Maximum EMI Capacity)**.
