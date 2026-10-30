"""
Maximum Monthly EMI Capacity Prediction View: Continuous Regressor.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
import streamlit as st

from app.components import (
    metric_card_html,
    floor_policy_callout
)
from app.validation import validate_applicant_inputs, calculate_financial_ratios
from app.db import save_prediction
from src.features.pipeline import FEATURE_NAMES_APPLICATION

FLOOR_POLICY_LIMIT = 500.0


def render_capacity_view(reg_pipeline):
    """Renders the continuous EMI borrowing limit estimation interface."""
    st.markdown('<div class="brand-title" style="font-size: 2rem; font-weight: 700;">'
                'Maximum Monthly EMI <span class="brand-gradient">Capacity Predictor</span>'
                '</div>', unsafe_allow_html=True)
    st.caption("Estimates the maximum safe monthly installment an applicant's balance sheet can sustain.")

    # Preset Profiles for Rapid Testing
    preset_choice = st.selectbox(
        "⚡ Quick-Fill Preset Financial Profiles (Optional):",
        ["Custom Manual Entry", "Profile 1: High Net Worth Tech Lead", "Profile 2: Mid-Income Stable Household", "Profile 3: Debt-Strained Minimal Cash Flow"],
        index=0,
        key="cap_preset"
    )

    defaults = {
        "gender": "Female", "marital_status": "Married", "education": "Post Graduate",
        "employment_type": "Private", "company_type": "Large Indian", "house_type": "Own",
        "existing_loans": "No", "emi_scenario": "Vehicle EMI",
        "age": 38, "salary": 85000, "years_emp": 8.0, "rent": 0,
        "dependents": 2, "school_fees": 6000, "college_fees": 0,
        "travel": 6000, "groceries": 15000, "other_exp": 7000,
        "current_emi": 0, "credit_score": 780, "bank_bal": 450000,
        "emerg_fund": 200000, "req_amount": 400000, "req_tenure": 36
    }

    if preset_choice == "Profile 1: High Net Worth Tech Lead":
        defaults.update({
            "salary": 180000, "rent": 0, "house_type": "Own", "dependents": 1,
            "school_fees": 8000, "groceries": 20000, "other_exp": 10000,
            "credit_score": 830, "bank_bal": 900000, "emerg_fund": 400000,
            "req_amount": 800000, "req_tenure": 48
        })
    elif preset_choice == "Profile 2: Mid-Income Stable Household":
        defaults.update({
            "salary": 55000, "rent": 10000, "house_type": "Rented", "dependents": 2,
            "school_fees": 4000, "groceries": 12000, "other_exp": 5000,
            "existing_loans": "Yes", "current_emi": 6000, "credit_score": 710,
            "bank_bal": 120000, "emerg_fund": 50000, "req_amount": 200000, "req_tenure": 24
        })
    elif preset_choice == "Profile 3: Debt-Strained Minimal Cash Flow":
        defaults.update({
            "salary": 28000, "rent": 9000, "house_type": "Rented", "dependents": 3,
            "school_fees": 4000, "groceries": 10000, "other_exp": 4000,
            "existing_loans": "Yes", "current_emi": 11000, "credit_score": 620,
            "bank_bal": 12000, "emerg_fund": 3000, "req_amount": 150000, "req_tenure": 12
        })

    with st.form("capacity_form"):
        st.markdown("#### 1. Applicant Financial Profile & Obligations")
        f_c1, f_c2, f_c3, f_c4 = st.columns(4)
        with f_c1:
            salary = st.number_input("Monthly Salary (₹)", min_value=1000, max_value=2000000, value=int(defaults["salary"]), step=5000)
            age = st.number_input("Age (Years)", min_value=18, max_value=85, value=int(defaults["age"]), step=1)
        with f_c2:
            house_type = st.selectbox("House Type", ["Own", "Family", "Rented"],
                                      index=["Own", "Family", "Rented"].index(defaults["house_type"]))
            rent = st.number_input("Monthly Rent (₹)", min_value=0, max_value=300000, value=int(defaults["rent"]), step=1000)
        with f_c3:
            years_emp = st.number_input("Experience (Years)", min_value=0.0, max_value=50.0,
                                        value=float(defaults["years_emp"]), step=0.5)
            dependents = st.number_input("Dependents", min_value=0, max_value=10, value=int(defaults["dependents"]), step=1)
        with f_c4:
            existing_loans = st.selectbox("Existing Loans?", ["No", "Yes"], index=0 if defaults["existing_loans"] == "No" else 1)
            current_emi = st.number_input("Current EMI Obligation (₹)", min_value=0, max_value=500000,
                                          value=int(defaults["current_emi"]), step=1000)

        st.markdown("#### 2. Living Expenses & Liquid Reserves")
        e_c1, e_c2, e_c3, e_c4 = st.columns(4)
        with e_c1:
            school_fees = st.number_input("School Fees (₹)", min_value=0, max_value=100000, value=int(defaults["school_fees"]), step=500)
            college_fees = st.number_input("College Tuition (₹)", min_value=0, max_value=200000, value=int(defaults["college_fees"]), step=1000)
        with e_c2:
            travel = st.number_input("Travel / Commute (₹)", min_value=0, max_value=100000, value=int(defaults["travel"]), step=500)
            groceries = st.number_input("Groceries & Utilities (₹)", min_value=500, max_value=200000, value=int(defaults["groceries"]), step=1000)
        with e_c3:
            other_exp = st.number_input("Other Living Expenses (₹)", min_value=0, max_value=200000, value=int(defaults["other_exp"]), step=500)
            credit_score = st.number_input("Bureau Credit Score", min_value=0, max_value=900, value=int(defaults["credit_score"]), step=10)
        with e_c4:
            bank_bal = st.number_input("Bank Balance (₹)", min_value=0, max_value=10000000, value=int(defaults["bank_bal"]), step=10000)
            emerg_fund = st.number_input("Emergency Fund (₹)", min_value=0, max_value=5000000, value=int(defaults["emerg_fund"]), step=5000)

        st.markdown("#### 3. Loan Application Context")
        l_c1, l_c2, l_c3 = st.columns(3)
        with l_c1:
            req_amount = st.number_input("Loan Amount (₹)", min_value=5000, max_value=5000000, value=int(defaults["req_amount"]), step=10000)
        with l_c2:
            req_tenure = st.number_input("Loan Tenure (Months)", min_value=3, max_value=84, value=int(defaults["req_tenure"]), step=3)
        with l_c3:
            scenario_opts = ["Vehicle EMI", "Personal Loan EMI", "Education EMI", "Home Appliances EMI", "E-commerce Shopping EMI"]
            emi_scenario = st.selectbox("Loan Purpose", scenario_opts, index=scenario_opts.index(defaults["emi_scenario"]))

        # Demographics defaults
        gender = defaults.get("gender", "Female")
        marital = defaults.get("marital_status", "Married")
        education = defaults.get("education", "Post Graduate")
        emp_type = defaults.get("employment_type", "Private")
        company_type = defaults.get("company_type", "Large Indian")

        submit_btn = st.form_submit_button("📐 Calculate Maximum Monthly EMI", use_container_width=True, type="primary")

    if submit_btn:
        applicant_payload = {
            "gender": gender, "marital_status": marital, "education": education,
            "employment_type": emp_type, "company_type": company_type, "house_type": house_type,
            "existing_loans": existing_loans, "emi_scenario": emi_scenario,
            "age": float(age), "monthly_salary": float(salary), "years_of_employment": float(years_emp),
            "monthly_rent": float(rent), "dependents": float(dependents), "school_fees": float(school_fees),
            "college_fees": float(college_fees), "travel_expenses": float(travel),
            "groceries_utilities": float(groceries), "other_monthly_expenses": float(other_exp),
            "current_emi_amount": float(current_emi if existing_loans == "Yes" else 0.0),
            "credit_score": float(credit_score), "bank_balance": float(bank_bal),
            "emergency_fund": float(emerg_fund), "requested_amount": float(req_amount),
            "requested_tenure": float(req_tenure)
        }

        is_valid, errors, warnings = validate_applicant_inputs(applicant_payload)
        for warn in warnings:
            st.warning(f"⚠️ {warn}")

        if not is_valid:
            st.error("❌ Submission blocked due to input validation errors:")
            for err in errors:
                st.error(f"• {err}")
            return

        input_df = pd.DataFrame([{col: applicant_payload[col] for col in FEATURE_NAMES_APPLICATION}])

        with st.spinner("Executing XGBoost Regressor Pipeline..."):
            raw_pred = float(reg_pipeline.predict(input_df)[0])
            policy_pred = max(FLOOR_POLICY_LIMIT, raw_pred)

        ratios = calculate_financial_ratios(applicant_payload)
        capacity_margin = policy_pred - ratios["implied_installment"]
        max_affordable_ticket = policy_pred * float(req_tenure)

        # Save to SQLite History
        rec_id = save_prediction(
            assessment_type="Capacity",
            applicant_data=applicant_payload,
            ratios=ratios,
            max_emi_raw=raw_pred,
            max_emi_policy=policy_pred,
            capacity_margin=capacity_margin,
            notes=f"Predicted raw capacity: ₹{raw_pred:,.2f}, Policy limit: ₹{policy_pred:,.2f}"
        )

        st.markdown("---")
        st.markdown("### 🏆 Estimated Borrowing Capacity Results")

        res_c1, res_c2, res_c3 = st.columns(3)
        with res_c1:
            st.markdown(metric_card_html(
                "Policy Monthly EMI Limit",
                f"₹{policy_pred:,.2f}",
                "Dataset-derived policy floor (₹500 threshold)"
            ), unsafe_allow_html=True)
        with res_c2:
            st.markdown(metric_card_html(
                "Raw Model Prediction",
                f"₹{raw_pred:,.2f}",
                "Unconstrained ML diagnostic output"
            ), unsafe_allow_html=True)
        with res_c3:
            st.markdown(metric_card_html(
                "Max Affordable Loan Size",
                f"₹{max_affordable_ticket:,.2f}",
                f"For tenure of {int(req_tenure)} months"
            ), unsafe_allow_html=True)

        if raw_pred < FLOOR_POLICY_LIMIT:
            floor_policy_callout(raw_pred, policy_pred)

        # Affordability Comparison
        st.markdown("#### ⚖️ Affordability Comparison: Requested Installment vs. Capacity")
        comp_c1, comp_c2 = st.columns(2)
        with comp_c1:
            st.write(f"• **Requested Monthly Installment (P/N):** ₹{ratios['implied_installment']:,.2f} *(assumes 0% nominal interest & zero fees)*")
            st.write(f"• **Approved Policy Monthly Limit:** ₹{policy_pred:,.2f} *(bounded at ₹500 dataset-derived policy floor)*")
            if capacity_margin >= 0:
                st.success(f"✓ **Capacity Surplus:** +₹{capacity_margin:,.2f} / month headroom available.")
            else:
                st.error(f"✕ **Capacity Deficit:** -₹{abs(capacity_margin):,.2f} / month over-budget.")

        with comp_c2:
            st.write(f"• **Declared Monthly Salary:** ₹{salary:,.2f}")
            st.write(f"• **Total Living & Debt Obligations:** ₹{ratios['total_obligations']:,.2f}")
            st.write(f"• **Net Uncommitted Disposable Income:** ₹{ratios['disposable_income']:,.2f}")

        st.caption(f"Audit Reference ID: #{rec_id} | Stored in local compliance ledger.")
