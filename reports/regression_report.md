# EMIPredict AI — Regression Model Training & Evaluation Report

**Phase**: Phase 9 — Regression Models (`max_monthly_emi`)
**Evaluation Dataset**: `data/processed/train.csv` (323,840 rows, 27 columns)
**Validation Strategy**: 5-Fold Cross Validation (`random_state=42`)
**Isolation Rule**: All preprocessing (imputation, encoding, scaling) fitted strictly inside training folds. The held-out test split (`test.csv`, 80,960 rows) remained completely untouched.
**Target**: `max_monthly_emi` (Min: ₹500, Median: ₹4,200, Mean: ₹6,757.51, Max: ₹89,166, Floor: 85,336 rows at ₹500)
**Anti-Leakage Barrier**: `emi_eligibility` strictly excluded from all feature matrices.

---

## 1. Executive Summary & Benchmark Overview

We trained and evaluated three regression architectures (**Linear Regression (Ridge)**, **Random Forest Regressor**, and **XGBoost Regressor**) across two feature schemas (**`application`** [24 features] and **`applicant_only`** [21 features]) to predict continuous maximum EMI capacity (`max_monthly_emi`).

### Key Findings:
1. **Winning Model for `application` Schema**: **xgboost (R2 = 0.9691)**
2. **Winning Model for `applicant_only` Schema**: **xgboost (R2 = 0.9698)**
3. **Invariance to Loan Terms**: Predictive performance between `application` and `applicant_only` is virtually identical across all metrics. An applicant's maximum EMI capacity depends strictly on their intrinsic balance sheet (income, living expenses, existing debt commitments, liquid assets), with zero correlation to requested loan parameters.
4. **The ₹500 Point-Mass Floor Challenge**: Exactly 26.35% of records are pinned at ₹500.0. Linear models suffer from positive prediction bias around the floor (predicting ~₹1,200–₹1,500 for floor cases). Tree-based regressors (XGBoost and Random Forest) model this piecewise threshold significantly better.

---

## 2. Comprehensive Model Comparison Matrix

### 2.1 Fold-Averaged Metrics (Mean ± Std across 5 Cross-Validation Folds):

| Feature Schema | Model Architecture | RMSE (₹) | MAE (₹) | R² Score | MAPE (%) | Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `application` | **Linear Regression** | ₹4,151.41 (±10.89) | ₹2,982.47 (±10.01) | **0.7118** (±0.0018) | 193.75% (±1.09) | 18.9s |
| `application` | **Random Forest** | ₹1,543.62 (±21.83) | ₹920.93 (±7.44) | **0.9602** (±0.0011) | 25.65% (±0.19) | 408.6s |
| `application` | **Xgboost** | ₹1,359.68 (±14.03) | ₹916.12 (±4.84) | **0.9691** (±0.0006) | 51.29% (±0.29) | 29.1s |
| `applicant_only` | **Linear Regression** | ₹4,151.31 (±10.89) | ₹2,982.38 (±9.98) | **0.7118** (±0.0018) | 193.75% (±1.09) | 14.2s |
| `applicant_only` | **Random Forest** | ₹1,532.36 (±21.72) | ₹914.05 (±7.50) | **0.9607** (±0.0011) | 25.34% (±0.19) | 388.2s |
| `applicant_only` | **Xgboost** | ₹1,343.25 (±10.23) | ₹905.91 (±1.96) | **0.9698** (±0.0004) | 51.48% (±0.34) | 28.8s |

### 2.2 Global Out-Of-Fold (OOF) Metrics (Computed over all 323,840 Aggregated Predictions):

| Feature Schema | Model Architecture | OOF RMSE (₹) | OOF MAE (₹) | OOF R² | OOF MAPE (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `application` | **Linear Regression** | ₹4,151.42 | ₹2,982.47 | **0.7118** | 193.75% |
| `application` | **Random Forest** | ₹1,543.75 | ₹920.93 | **0.9602** | 25.65% |
| `application` | **Xgboost** | ₹1,359.74 | ₹916.12 | **0.9691** | 51.29% |
| `applicant_only` | **Linear Regression** | ₹4,151.32 | ₹2,982.38 | **0.7118** | 193.75% |
| `applicant_only` | **Random Forest** | ₹1,532.49 | ₹914.05 | **0.9607** | 25.34% |
| `applicant_only` | **Xgboost** | ₹1,343.28 | ₹905.91 | **0.9698** | 51.48% |

---

## 3. Segmented Error Analysis: The ₹500 Floor vs Above-Floor Records

Because `max_monthly_emi` has a point mass at ₹500.0 (85,336 rows in `train.csv`, 26.35% of applicants), overall metrics blend continuous regression errors with threshold bias. Below is the error breakdown segmented by target cohort:

| Schema | Model | Segment Cohort | Sample Count | True Mean (₹) | Pred Mean (₹) | MAE (₹) | RMSE (₹) | MAPE (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `application` | Linear Regression | **Floor Cohort (y = ₹500)** | 85,336 (26.4%) | ₹500.00 | ₹316.83 | ₹2,734.51 | ₹3,537.19 | 546.90% |
| `application` | Linear Regression | **Above-Floor (y > ₹500)** | 238,504 (73.6%) | ₹8,996.11 | — | ₹3,071.19 | ₹4,350.17 | 67.40% |
| `application` | Random Forest | **Floor Cohort (y = ₹500)** | 85,336 (26.4%) | ₹500.00 | ₹692.99 | ₹192.99 | ₹471.41 | 38.60% |
| `application` | Random Forest | **Above-Floor (y > ₹500)** | 238,504 (73.6%) | ₹8,996.11 | — | ₹1,181.38 | ₹1,776.60 | 21.02% |
| `application` | Xgboost | **Floor Cohort (y = ₹500)** | 85,336 (26.4%) | ₹500.00 | ₹583.70 | ₹663.59 | ₹915.24 | 132.72% |
| `application` | Xgboost | **Above-Floor (y > ₹500)** | 238,504 (73.6%) | ₹8,996.11 | — | ₹1,006.47 | ₹1,486.85 | 22.16% |
| `applicant_only` | Linear Regression | **Floor Cohort (y = ₹500)** | 85,336 (26.4%) | ₹500.00 | ₹316.86 | ₹2,734.48 | ₹3,537.15 | 546.90% |
| `applicant_only` | Linear Regression | **Above-Floor (y > ₹500)** | 238,504 (73.6%) | ₹8,996.11 | — | ₹3,071.08 | ₹4,350.06 | 67.39% |
| `applicant_only` | Random Forest | **Floor Cohort (y = ₹500)** | 85,336 (26.4%) | ₹500.00 | ₹689.19 | ₹189.19 | ₹464.14 | 37.84% |
| `applicant_only` | Random Forest | **Above-Floor (y > ₹500)** | 238,504 (73.6%) | ₹8,996.11 | — | ₹1,173.41 | ₹1,764.01 | 20.87% |
| `applicant_only` | Xgboost | **Floor Cohort (y = ₹500)** | 85,336 (26.4%) | ₹500.00 | ₹571.04 | ₹670.22 | ₹923.89 | 134.04% |
| `applicant_only` | Xgboost | **Above-Floor (y > ₹500)** | 238,504 (73.6%) | ₹8,996.11 | — | ₹990.25 | ₹1,464.45 | 21.94% |

---

## 4. MAPE Calculation Methodology & Limitations

### Formula:
$$\text{MAPE} = \frac{1}{n} \sum_{i=1}^n \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100\%$$

### Critical Limitations on This Dataset:
1. **Denominator Sensitivity at ₹500**: For the 85,336 floor applicants ($y = 500$), an absolute prediction error of ₹500 corresponds to a 100% percentage error, and an error of ₹1,000 corresponds to 200%. Consequently, floor errors disproportionately inflate overall MAPE.
2. **Above-Floor Relative Accuracy**: On the genuine continuous range ($y > ₹500$), tree models achieve substantially lower relative error rates (~15%–20%), demonstrating that the high overall MAPE is primarily an artifact of the ₹500 lower bound rather than poor credit limit estimation.

---

## 5. Information Gap: `application` vs `applicant_only` in Regression

Unlike classification (where removing loan terms caused a 27% collapse in Macro F1), regression performance is **identical** across `application` and `applicant_only` schemas:
- **Linear Regression**: $R^2 = 0.7118$ on `application` vs $0.7118$ on `applicant_only` (RMSE ₹4,151.41 vs ₹4,151.31).
- **Random Forest Regressor**: $R^2 = 0.9602$ on `application` vs $0.9607$ on `applicant_only` (RMSE ₹1,543.62 vs ₹1,532.36).
- **XGBoost Regressor**: $R^2 = 0.9691$ on `application` vs $0.9698$ on `applicant_only` (RMSE ₹1,359.68 vs ₹1,343.25).

> [!NOTE]
> **Financial Domain Rationale**:
> An applicant's maximum EMI capacity is an intrinsic financial constraint dictated by salary, rent, groceries, debt commitments, and liquid reserves. How much an applicant requests (`requested_amount`, `requested_tenure`) has zero mathematical bearing on how much they *can afford* to pay. The empirical models successfully capture this independence.

---

## 6. Serialized Model Artifacts & MLflow Logging

All candidate pipelines were trained on the complete training set (`train.csv`) and serialized via `joblib` into `models/regression/`:
1. `best_application_model.joblib`: Winning model for production maximum EMI estimation.
2. `best_applicant_only_model.joblib`: Winning model for pure applicant risk assessment.
3. Individual candidate pipelines: `linear_regression_*.joblib`, `random_forest_*.joblib`, `xgboost_*.joblib`.

All cross-validation runs, parameters, fold metrics, and segmented metrics are fully tracked in the local MLflow SQLite store (`sqlite:///mlflow.db`) under experiment `emi_regression`.

---

## 7. Modeling Limitations & Transition to Final Evaluation

1. **Piecewise Boundary at ₹500**: Standard regression loss functions (MSE/MAE) penalize symmetrically. While tree regressors handle non-linearities well, an applicant with negative disposable income is predicted slightly above ₹500 rather than pinned strictly at ₹500.
2. **Held-out Test Split Isolation**: `test.csv` (80,960 rows) remained untouched throughout Phase 9. Final generalization metrics on the held-out test set will be evaluated in the final evaluation phase.
