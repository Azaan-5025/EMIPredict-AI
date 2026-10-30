"""
Dashboard View: Executive Platform Overview & Real-Time Underwriting KPI Monitor.
"""

import streamlit as st
from app.components import (
    metric_card_html,
    synthetic_disclaimer_callout
)
from app.db import get_history_summary_stats


def render_dashboard_view():
    """Renders the executive overview dashboard."""
    st.markdown('<div class="brand-title" style="font-size: 2.2rem; font-weight: 700; margin-bottom: 4px;">'
                '<span class="brand-gradient">EMIPredict AI</span> Platform Dashboard'
                '</div>', unsafe_allow_html=True)
    st.caption("Intelligent Multi-Task Financial Risk Assessment & EMI Capacity Underwriting Engine")

    synthetic_disclaimer_callout()

    st.markdown("### 📊 Production Model Benchmarks (Held-Out Test Set: N = 80,960)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card_html("Macro F1-Score", "0.6968", "XGBoost Balanced (Test)"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card_html("ROC-AUC (OvR)", "0.9620", "Multi-Class Discrimination"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card_html("Regression R²", "0.9712", "Clipped at ₹500 Floor"), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card_html("Generalization Gap", "< 0.36%", "5-Fold CV vs Test Split"), unsafe_allow_html=True)

    st.markdown("---")

    # Historical Activity Metrics
    stats = get_history_summary_stats()
    st.markdown("### 📈 Real-Time Platform Audit Telemetry")
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    with col_t1:
        st.metric(label="Total Audited Applications", value=f"{stats['total_records']:,}")
    with col_t2:
        st.metric(label="Average Approved Capacity", value=f"₹{stats['average_max_emi']:,.2f}")
    with col_t3:
        st.metric(label="Average Portfolio FOIR", value=f"{stats['average_foir']:.1f}%")
    with col_t4:
        st.metric(label="Average Loan Request", value=f"₹{stats['average_requested_amount']:,.2f}")

    st.markdown("---")

    # Interactive Underwriting Simulator
    st.markdown("### ⚡ Interactive Underwriting Capacity Simulator")
    st.markdown("Quickly test how applicant cash flow and proposed installment interact under Indian banking FOIR thresholds:")

    sim_c1, sim_c2 = st.columns(2)
    with sim_c1:
        sim_salary = st.slider("Monthly Net Salary (₹)", min_value=15000, max_value=250000, value=65000, step=5000)
        sim_existing_emi = st.slider("Current Monthly Debt EMIs (₹)", min_value=0, max_value=100000, value=8000, step=2000)
        sim_living_exp = st.slider("Monthly Living Expenses (Rent, Utilities, Food) (₹)", min_value=5000, max_value=120000, value=25000, step=2500)

    with sim_c2:
        sim_loan = st.slider("Proposed Loan Ticket Size (₹)", min_value=20000, max_value=1000000, value=200000, step=10000)
        sim_tenure = st.slider("Proposed Tenure (Months)", min_value=6, max_value=60, value=24, step=3)

        sim_installment = sim_loan / sim_tenure
        sim_total_exp = sim_existing_emi + sim_living_exp
        sim_disposable = sim_salary - sim_total_exp
        sim_projected_foir = ((sim_existing_emi + sim_installment) / sim_salary) * 100.0

        st.markdown(f"**Implied Monthly Installment (P/N):** ₹{sim_installment:,.2f} *(assumes 0% nominal interest & zero fees)*")
        st.markdown(f"**Remaining Disposable Cash Flow:** ₹{sim_disposable:,.2f}")

        if sim_projected_foir <= 40.0:
            st.success(f"Projected FOIR: {sim_projected_foir:.1f}% — Safe Leverage Tier (≤ 40%)")
        elif sim_projected_foir <= 55.0:
            st.warning(f"Projected FOIR: {sim_projected_foir:.1f}% — Moderate / Borderline Leverage (40% - 55%)")
        else:
            st.error(f"Projected FOIR: {sim_projected_foir:.1f}% — High Risk Over-Leveraged (> 55%)")

    st.markdown("---")

    # Dual Pipeline Architecture Overview
    st.markdown("### 🛡️ Production Model Architecture & Safeguards")
    arch_c1, arch_c2 = st.columns(2)
    with arch_c1:
        st.markdown("""
        **1. Eligibility Classifier (`application` schema)**:
        - **Model**: Gradient Boosted Trees (XGBoost) with balanced class reweighting.
        - **Features (24)**: Full applicant balance-sheet + loan request terms (`requested_amount`, `requested_tenure`, `emi_scenario`).
        - **Protection**: `max_monthly_emi` is programmatically barred to avoid trivial label leakage.
        - **Performance**: 80.93% recall on minority `High_Risk` applicants on the test set.
        """)
    with arch_c2:
        st.markdown("""
        **2. Maximum Monthly EMI Regressor (`application` schema)**:
        - **Model**: XGBoost Regressor with histogram binning.
        - **Target**: Continuous repayment capacity limit (`max_monthly_emi`).
        - **Protection**: `emi_eligibility` is strictly barred from all inputs.
        - **Policy Enforcement**: Automatic diagnostic distinction between raw model output and the ₹500/month policy floor.
        """)
