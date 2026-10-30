"""
Model Governance & Benchmark View: Training CV vs. Held-Out Test Evaluation.
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd

from app.components import (
    metric_card_html,
    synthetic_disclaimer_callout
)


def render_model_performance_view():
    """Renders comprehensive model governance, benchmarking, and synthetic data audit documentation."""
    st.markdown('<div class="brand-title" style="font-size: 2rem; font-weight: 700;">'
                'Model Governance & <span class="brand-gradient">Benchmark Audit</span>'
                '</div>', unsafe_allow_html=True)
    st.caption("Transparent audit of 5-Fold Cross-Validation, Held-Out Test Benchmarks, Generalization Gaps, and Anti-Leakage Architecture.")

    synthetic_disclaimer_callout()

    tabs = st.tabs([
        "1. Classification Benchmarks",
        "2. Regression Benchmarks",
        "3. Segmented Floor Analysis",
        "4. Multi-Task Alignment",
        "5. Critical Limitations of Synthetic Data"
    ])

    # Tab 1: Classification
    with tabs[0]:
        st.markdown("### 🎯 Classification Benchmarks: CV vs. Held-Out Test (`test.csv`, N = 80,960)")
        st.markdown("Evaluated on the frozen held-out test split using `models/classification/best_application_model.joblib` (XGBoost Balanced) and candidate models:")

        clf_data = [
            {"Model Pipeline": "XGBoost (Balanced) [WINNER]", "Schema": "application", "5-Fold CV F1": "0.6953", "Test F1": "0.6968", "Gap (Δ)": "+0.0015 (+0.22%)", "Test Bal Acc": "0.8220", "Test ROC-AUC": "0.9620", "High_Risk Recall": "80.93%"},
            {"Model Pipeline": "Random Forest (Balanced) [BASELINE]", "Schema": "applicant_only", "5-Fold CV F1": "0.5037", "Test F1": "0.5016", "Gap (Δ)": "-0.0021 (-0.42%)", "Test Bal Acc": "0.5457", "Test ROC-AUC": "0.8205", "High_Risk Recall": "20.19%"},
            {"Model Pipeline": "Logistic Regression (Balanced)", "Schema": "application", "5-Fold CV F1": "0.6276", "Test F1": "0.6298", "Gap (Δ)": "+0.0022 (+0.35%)", "Test Bal Acc": "0.7307", "Test ROC-AUC": "0.9224", "High_Risk Recall": "65.31%"},
            {"Model Pipeline": "Random Forest (Balanced)", "Schema": "application", "5-Fold CV F1": "0.6685", "Test F1": "0.6721", "Gap (Δ)": "+0.0036 (+0.54%)", "Test Bal Acc": "0.7510", "Test ROC-AUC": "0.9425", "High_Risk Recall": "57.02%"}
        ]
        st.dataframe(pd.DataFrame(clf_data), use_container_width=True, hide_index=True)

        st.markdown("""
        **Key Classification Insights**:
        - **Generalization Gap < 0.36%**: The difference between 5-fold cross-validation and held-out test performance is negligible across all models, proving that transformers and encoders did not leak training data.
        - **Importance of Balanced Weighting**: Without balanced class weighting, models suffer catastrophic collapse on the 4.32% `High_Risk` minority (0.00% recall). Balanced weighting achieves **80.93% recall** on test `High_Risk` applicants.
        - **Information Gap**: Withholding loan request terms (`requested_amount`, `requested_tenure`) drops Macro F1 from 0.6968 down to 0.5016, proving that credit risk inherently depends on loan ticket size relative to capacity.
        """)

    # Tab 2: Regression
    with tabs[1]:
        st.markdown("### 📐 Regression Benchmarks: Maximum Monthly EMI (`max_monthly_emi`)")
        st.markdown("Comparing unconstrained raw predictions against predictions clipped at the ₹500 institutional policy floor:")

        reg_data = [
            {"Architecture": "XGBoost Regressor [WINNER]", "Schema": "application", "Raw R²": "0.9698", "Clipped R²": "0.9712", "Raw RMSE": "₹1,350.46", "Clipped RMSE": "₹1,319.46", "Raw MAE": "₹913.68", "Clipped MAE": "₹838.55", "MAPE": "50.84% (35.84% clip)"},
            {"Architecture": "XGBoost Regressor [BASELINE]", "Schema": "applicant_only", "Raw R²": "0.9706", "Clipped R²": "0.9721", "Raw RMSE": "₹1,331.56", "Clipped RMSE": "₹1,297.84", "Raw MAE": "₹897.76", "Clipped MAE": "₹818.07", "MAPE": "51.11% (35.19% clip)"},
            {"Architecture": "Random Forest Regressor", "Schema": "application", "Raw R²": "0.9619", "Clipped R²": "0.9619", "Raw RMSE": "₹1,516.84", "Clipped RMSE": "₹1,516.84", "Raw MAE": "₹911.13", "Clipped MAE": "₹911.13", "MAPE": "25.23%"},
            {"Architecture": "Linear Regression (Ridge)", "Schema": "application", "Raw R²": "0.7138", "Clipped R²": "0.7463", "Raw RMSE": "₹4,158.24", "Clipped RMSE": "₹3,915.17", "Raw MAE": "₹2,997.64", "Clipped MAE": "₹2,584.60", "MAPE": "194.54% (115.4% clip)"}
        ]
        st.dataframe(pd.DataFrame(reg_data), use_container_width=True, hide_index=True)

        st.markdown("""
        **Loan Term Invariance in Regression**:
        - Unlike classification, regression model performance is identical across `application` and `applicant_only` ($R^2 \approx 0.970$).
        - **Financial Logic**: Maximum repayment capacity depends exclusively on an applicant's intrinsic cash flow (salary, living costs, existing debt, savings), with zero mathematical dependence on how much money the applicant asks to borrow.
        """)

    # Tab 3: Segmented Floor Analysis
    with tabs[2]:
        st.markdown("### 🧱 The ₹500 Point-Mass Floor: Segmented Error Analysis")
        st.markdown("The test split contains **21,347 records (26.37%)** clamped at ₹500 and **59,613 records (73.63%)** above ₹500 (True Mean = ₹9,039.63):")

        floor_data = [
            {"Segment Cohort": "Floor Cohort (y = ₹500)", "Model": "XGBoost Regressor", "Count": "21,347 (26.4%)", "True Mean": "₹500.00", "Raw Pred Mean": "₹591.01", "Raw MAE": "₹658.07", "Clipped MAE": "₹374.54", "Raw RMSE": "₹914.72", "Clipped RMSE": "₹724.63"},
            {"Segment Cohort": "Above-Floor (y > ₹500)", "Model": "XGBoost Regressor", "Count": "59,613 (73.6%)", "True Mean": "₹9,039.63", "Raw Pred Mean": "₹9,027.84", "Raw MAE": "₹1,005.22", "Clipped MAE": "₹1,004.71", "Raw RMSE": "₹1,475.53", "Clipped RMSE": "₹1,475.25"},
            {"Segment Cohort": "Floor Cohort (y = ₹500)", "Model": "Linear Regression (Ridge)", "Count": "21,347 (26.4%)", "True Mean": "₹500.00", "Raw Pred Mean": "₹330.55", "Raw MAE": "₹2,750.11", "Clipped MAE": "₹1,290.33", "Raw RMSE": "₹3,555.67", "Clipped RMSE": "₹2,418.47"},
            {"Segment Cohort": "Above-Floor (y > ₹500)", "Model": "Linear Regression (Ridge)", "Count": "59,613 (73.6%)", "True Mean": "₹9,039.63", "Raw Pred Mean": "₹9,111.88", "Raw MAE": "₹3,086.28", "Clipped MAE": "₹3,048.06", "Raw RMSE": "₹4,353.79", "Clipped RMSE": "₹4,327.02"}
        ]
        st.dataframe(pd.DataFrame(floor_data), use_container_width=True, hide_index=True)

        st.markdown("""
        **Why MAPE Behaves Poorly on Floor Cohorts**:
        - For the 21,347 floor applicants ($y = 500$), an absolute error of just ₹500 represents a 100% percentage error. This inflates overall MAPE to ~50%.
        - On the genuine continuous range ($y > ₹500$), XGBoost achieves a **21.86% MAPE**, demonstrating strong relative limit accuracy across middle- and high-income applicants.
        """)

    # Tab 4: Multi-Task Alignment
    with tabs[3]:
        st.markdown("### 🔄 Multi-Task Alignment: Consistency Between Regression & Classification")
        st.markdown("We audited cross-model coherence on all 80,960 held-out test records:")

        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.markdown(metric_card_html("Debt Ratio > 1.0 Agreement", "89.14%", "Classified as Not_Eligible"), unsafe_allow_html=True)
            st.caption("When implied installment exceeds predicted maximum capacity, 89.14% are independently rejected.")
        with col_a2:
            st.markdown(metric_card_html("Debt Ratio ≤ 0.526 Agreement", "93.70%", "Classified as Eligible"), unsafe_allow_html=True)
            st.caption("When implied installment has ample headroom, 93.70% are independently approved.")

        st.markdown(r"""
        - **Borderline Cohort**: Applicants classified as `High_Risk` had a median predicted debt ratio of **0.938** (IQR: 0.726–1.214), precisely matching the ground-truth synthetic borderline zone ($0.777 < Ratio \le 0.974$).
        """)

    # Tab 5: Critical Limitations
    with tabs[4]:
        st.markdown("### ⚠️ Methodological Notice: Limitations of Synthetic Data in Banking")
        st.error(r"""
        **DO NOT CONFUSE MATHEMATICAL REVERSE-ENGINEERING WITH REAL-WORLD BANKING VALIDITY:**

        1. **Deterministic Generator Rules**: The underlying dataset generated labels via explicit algebraic rules:
           $$Ratio = \frac{\text{requested\_amount} / \text{requested\_tenure}}{\text{max\_monthly\_emi}}$$
           High machine learning performance reflects the model's ability to approximate these continuous piecewise rules, NOT an ability to predict human solvency.

        2. **Absence of Real Compounding Interest**: The dataset assumes $I_{req} = P / N$ (0% nominal APR). Real bank personal and vehicle loans carry 9% to 24% compounding APR, increasing real debt obligations by 8% to 25%.

        3. **Decoupled Credit Bureau Dynamics**: In the synthetic generator, credit score was completely decoupled from loan eligibility (~18.4% Eligible across all score brackets). In real underwriting, subprime bureau scores trigger automatic policy decline.

        4. **Stochastic Behavioral Shocks**: Real-world defaults stem from macroeconomic shocks (inflation, rate hikes), job displacement, illness, or divorce—none of which are represented in static synthetic tabular records.
        """)
