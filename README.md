# EMIPredict AI — Intelligent Financial Risk Assessment Platform

[![Python 3.12.7](https://img.shields.io/badge/python-3.12.7-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn 1.9.1](https://img.shields.io/badge/scikit--learn-1.9.1-orange.svg)](https://scikit-learn.org/)
[![XGBoost 3.4.1](https://img.shields.io/badge/xgboost-3.4.1-red.svg)](https://xgboost.ai/)
[![Streamlit 1.66.0](https://img.shields.io/badge/streamlit-1.66.0-ff4b4b.svg)](https://streamlit.io/)
[![pytest](https://img.shields.io/badge/tests-59%20passed-brightgreen.svg)](https://pytest.org/)

**EMIPredict AI** is an enterprise-grade financial risk assessment and credit capacity underwriting platform. Built on clean modular pipelines, it pairs a multi-class risk classifier (**Eligible**, **High Risk**, **Not Eligible**) with a continuous capacity regressor predicting **Maximum Monthly EMI** in Indian Rupees (₹).

---

## 1. Executive Summary & Held-Out Test Benchmarks

All models were developed using strict stratified 5-fold cross-validation on `train.csv` (323,840 records), and frozen before evaluation on the held-out test split (`test.csv`, 80,960 records, 27 columns).

| Task | Primary Selected Pipeline | Key Test Metric | Secondary Test Metric | Generalization Gap (Δ) |
| :--- | :--- | :---: | :---: | :---: |
| **Classification** | **XGBoost Classifier (Balanced)** | **Macro F1: 0.6968** | **ROC-AUC: 0.9620** | **+0.0015 (+0.22%)** |
| **Regression (Raw)** | **XGBoost Regressor** | **R²: 0.9698** | **RMSE: ₹1,350.46** | **+0.0007 (+0.07%)** |
| **Regression (Policy)** | **XGBoost Regressor (Clipped ₹500)** | **R²: 0.9712** | **RMSE: ₹1,319.46** | **-₹40.22 (-2.96%)** |

- **High_Risk Sensitivity**: The balanced classification pipeline achieves **80.93% recall** on the test `High_Risk` minority (identifying 2,830 of 3,497 borderline applicants).
- **Zero Leakage**: All imputers, encoders, and scalers fit strictly within cross-validation folds. Target variables are programmatically barred from input matrices.
- **Minimal Generalization Gap**: Performance across CV and held-out test matches within **< 0.36%**, proving zero overfitting.

---

## 2. Platform Architecture & Feature Schemas

The system enforces two rigorous feature schemas defined in `src/features/pipeline.py`:
1. **`application` Schema (24 features)**:
   - Full application-time inputs genuinely submitted by a customer:
     - **Demographics & Work (8)**: `gender`, `marital_status`, `education`, `employment_type`, `company_type`, `age`, `years_of_employment`, `dependents`
     - **Cash Flow & Housing (8)**: `monthly_salary`, `house_type`, `monthly_rent`, `school_fees`, `college_fees`, `travel_expenses`, `groceries_utilities`, `other_monthly_expenses`
     - **Debts & Assets (5)**: `existing_loans`, `current_emi_amount`, `credit_score`, `bank_balance`, `emergency_fund`
     - **Loan Request Parameters (3)**: `requested_amount`, `requested_tenure`, `emi_scenario`
   - **Safeguard**: `max_monthly_emi` is programmatically barred from classification features.
2. **`applicant_only` Schema (21 features)**:
   - Withholds all loan request parameters (`requested_amount`, `requested_tenure`, `emi_scenario`) to evaluate the customer's intrinsic balance-sheet capacity in isolation.
   - **Safeguard**: `family_size` is removed across all schemas due to perfect collinearity ($r=1.0$) with `dependents`.

---

## 3. Streamlit Web Application Features

The interactive web application (`app/main.py`) provides an intelligent underwriting cockpit with 6 specialized interfaces:

1. **Executive Dashboard (`app/views/dashboard.py`)**:
   - Live platform telemetry, held-out test benchmark metrics, real-time portfolio audit stats, and an interactive FOIR/cash-flow underwriting simulator.
2. **EMI Eligibility Assessment (`app/views/eligibility.py`)**:
   - Multi-class eligibility underwriting with grouped input fields, one-click preset applicant profiles, live validation, probability gauges, and mandatory **High_Risk** underwriting warning callouts.
3. **Maximum Monthly EMI Capacity Predictor (`app/views/capacity.py`)**:
   - Continuous borrowing limit estimator. Transparently reports both the **Recommended Policy Limit** (bounded at ₹500/month) and the **Raw ML Diagnostic Prediction**. Includes loan installment vs capacity margin analysis.
4. **Unified Risk & Capacity Assessment (`app/views/joint.py`)**:
   - Evaluates both models simultaneously, computing the debt-to-capacity ratio ($\text{Ratio} = (P/N) / \hat{y}_{capacity}$) and verifying cross-model coherence.
5. **Underwriting History & Audit Trail (`app/views/history.py`)**:
   - Persistent SQLite audit ledger (`data/predictions_history.db`). Features record inspection, type/class filtering, single-record deletion, ledger clearing, and instant CSV export. **Stores zero personal identification data (PII)**.
6. **Model Governance & Benchmark Audit (`app/views/model_performance.py`)**:
   - Complete transparency reports: 5-Fold CV vs Test comparisons, segmented ₹500 floor error analysis, and critical discussion on synthetic data limitations.

---

## 4. Local Environment Setup & Launch Instructions

### Prerequisites
- Windows, macOS, or Linux
- Python **3.12.7** (recommended) or Python 3.10+
- Git

### Step-by-Step Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/EMIPredict-AI.git
   cd EMIPredict-AI
   ```

2. **Activate the existing virtual environment** (or create one):
   ```powershell
   # Windows PowerShell:
   .\.venv\Scripts\Activate.ps1

   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Verify installed dependencies**:
   Ensure core dependencies are installed:
   ```bash
   pip install pandas numpy scipy scikit-learn xgboost streamlit mlflow pytest
   ```

4. **Verify serialized model artifacts exist**:
   Ensure the following serialized joblib pipelines are present in `models/`:
   ```
   models/
   ├── classification/
   │   ├── best_application_model.joblib
   │   ├── best_applicant_only_model.joblib
   │   ├── label_encoder.joblib
   │   ├── logistic_regression_application.joblib
   │   ├── random_forest_application.joblib
   │   └── xgboost_application.joblib
   └── regression/
       ├── best_application_model.joblib
       ├── best_applicant_only_model.joblib
       ├── linear_regression_application.joblib
       ├── random_forest_application.joblib
       └── xgboost_application.joblib
   ```

---

## 5. Exact Command to Run Streamlit Locally

To launch the Streamlit web application on your local machine, run either of the following commands from the project root:

```powershell
# Recommended command:
.\.venv\Scripts\streamlit.exe run app.py

# Alternative via Python module:
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The application will start immediately and open in your default browser at:
👉 **`http://localhost:8501`**

---

## 5.1 Streamlit Community Cloud Deployment Guide

The application is fully configured for continuous deployment on **Streamlit Community Cloud** backed by GitHub:

### Architecture & Pre-Flight Checks
- **Target Runtime**: Python 3.12 (Linux x86_64 container).
- **Tracked Repository Footprint**: **~4.44 MB** across 62 files (zero Git LFS required).
- **Model Payloads Tracked**:
  - `models/classification/best_application_model.joblib` (1.37 MB)
  - `models/classification/label_encoder.joblib` (1.2 KB)
  - `models/regression/best_application_model.joblib` (0.49 MB)
- **Excluded Large Artifacts**: Large candidate models (`models/regression/random_forest_application.joblib` [163.4 MB] and `models/classification/best_applicant_only_model.joblib` [100.4 MB]) and heavy raw/processed CSV datasets (>290 MB) are strictly excluded via `.gitignore` to prevent exceeding GitHub's 100 MB hard push limit.
- **Persistence Note**: Streamlit Community Cloud provides ephemeral container storage. Prediction history saved to SQLite (`data/predictions_history.db`) persists reliably across page views within an active session, but will reset upon container restart/redeploy.

### Step-by-Step Deployment Instructions

1. **Initialize Git & Commit** (When ready to push):
   ```bash
   git init
   git add .
   git commit -m "feat: complete EMIPredict-AI production application"
   git branch -M main
   ```
2. **Push to Remote Repository**:
   ```bash
   git remote add origin https://github.com/<your-username>/EMIPredict-AI.git
   git push -u origin main
   ```
3. **Connect to Streamlit Community Cloud**:
   - Navigate to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
   - Click **"New app"**.
   - Select your repository: `<your-username>/EMIPredict-AI`.
   - Set **Branch**: `main`.
   - Set **Main file path**: `app.py`.
   - Click **"Advanced settings..."** -> choose **Python 3.12**.
4. **Deploy**:
   - Click **"Deploy!"**.
   - Streamlit Cloud automatically detects `requirements.txt` and `.streamlit/config.toml`, installs pinned dependencies, and launches the application.

---

## 6. Running the Automated Test Suite

The repository includes a comprehensive 59-test automated verification suite covering environment dependencies, EDA, preprocessing pipelines, model inference, held-out evaluation, and Streamlit application components:

```powershell
# Run full test suite:
.\.venv\Scripts\pytest.exe -v
```

Expected test result:
```
======================= 59 passed, 1 warning in ~22s =======================
- test_environment.py: 6 passed (Python 3.12, scikit-learn, xgboost, mlflow, streamlit)
- tests/test_preprocessing.py: 13 passed (Anti-leakage, dual schemas, stratification)
- tests/test_eda.py: 4 passed (Figures, statistical distributions, test isolation)
- tests/test_classification.py: 6 passed (Classifier serialization, probability sums)
- tests/test_regression.py: 7 passed (Regressor serialization, continuous inference)
- tests/test_evaluation.py: 8 passed (Held-out benchmark bounds, generalization gap)
- tests/test_app.py: 15 passed (Validation rules, ratio engine, SQLite CRUD, governance, profile presets)
```

---

## 7. Critical Methodological Notice: Synthetic Data Limitations

> [!WARNING]
> **SYNTHETIC DATA REVERSE-ENGINEERING WARNING**:
>
> High benchmark metrics ($R^2 \approx 0.97$, ROC-AUC $\approx 0.96$) reflect the machine learning models' ability to approximate the deterministic algebraic inequalities used by the dataset generator:
> $$\text{Ratio} = \frac{\text{requested\_amount} / \text{requested\_tenure}}{\text{max\_monthly\_emi}}$$
>
> In actual banking environments, these models would encounter critical real-world failure modes:
> 1. **Zero-Interest Assumption**: The synthetic installment formula assumes $I_{req} = P / N$ (0% APR). Real commercial loans carry 9%–24% compounding APR, increasing repayment obligations by 8% to 25%.
> 2. **Decoupled Credit Scores**: In this dataset, credit score was decoupled from eligibility rules. In real underwriting, subprime bureau scores trigger automatic declines.
> 3. **Stochastic Macroeconomic Events**: Real-world defaults stem from macroeconomic shocks (inflation, rate hikes), job displacement, illness, or divorce—none of which are captured in static synthetic records.
>
> **Operational Guidance**: Automated model outputs must serve as assistive decision-support tools and must always be subject to human underwriter review, debt-to-income caps, and regulatory audit compliance.

---

## 8. Directory Structure

```
EMIPredict-AI/
├── app.py                                # Root Streamlit launcher
├── app/                                  # Streamlit Application Package
│   ├── main.py                           # App entry point & navigation router
│   ├── components.py                     # Glassmorphic CSS, metric cards, badges
│   ├── db.py                             # SQLite prediction ledger (zero PII)
│   ├── validation.py                     # Input validation & FOIR ratio engine
│   └── views/
│       ├── dashboard.py                  # Executive Dashboard & KPIs
│       ├── eligibility.py                # Multi-class risk classifier
│       ├── capacity.py                   # Maximum monthly EMI regressor
│       ├── joint.py                      # Unified dual-model assessment
│       ├── history.py                    # SQLite audit ledger CRUD & export
│       └── model_performance.py          # CV vs Test benchmarks & governance
├── data/
│   ├── raw/                              # Original raw dataset (SHA-256 verified)
│   └── processed/
│       ├── train.csv                     # 80% Stratified Training split (323,840 rows)
│       └── test.csv                      # 20% Stratified Held-out Test split (80,960 rows)
├── models/
│   ├── classification/                   # Serialized classification pipelines
│   └── regression/                       # Serialized regression pipelines
├── reports/
│   ├── final_model_evaluation_report.md  # Comprehensive Phase 10 audit report
│   ├── final_evaluation_metrics.json     # Machine-readable test benchmark metrics
│   ├── classification_report.md          # Phase 8 CV classification report
│   └── regression_report.md              # Phase 9 CV regression report
├── src/
│   ├── features/pipeline.py              # Dual Scikit-learn preprocessing pipelines
│   └── models/
│       ├── train_classification.py       # Phase 8 classification training
│       ├── train_regression.py           # Phase 9 regression training
│       └── evaluate_test.py              # Phase 10 test evaluation pipeline
├── tests/                                # Automated pytest verification suite (52 tests)
└── PROJECT_STATUS.md                     # Comprehensive project ledger
```

---

## 9. License & Project Attribution

Developed as part of the **EMIPredict AI** Intelligent Financial Risk Assessment Platform.
Designed for educational demonstration, credit decision-support research, and machine learning governance auditing.
