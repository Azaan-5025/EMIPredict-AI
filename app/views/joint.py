"""
Joint Multi-Task Assessment View: Unified Eligibility & Capacity Underwriting Engine.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
import streamlit as st

from app.components import (
    status_badge_html,
    render_probability_bars,
    high_risk_warning_callout,
    floor_policy_callout,
    metric_card_html
)
from app.validation import validate_applicant_inputs, calculate_financial_ratios
from app.db import save_prediction
from src.features.pipeline import FEATURE_NAMES_APPLICATION

FLOOR_POLICY_LIMIT = 500.0


def render_joint_view(clf_pipeline, reg_pipeline, le):
    """Renders the unified dual-model underwriting simulator."""
    st.markdown('<div class="brand-title" style="font-size: 2rem; font-weight: 700;">'
                'Unified Credit & Capacity <span class="brand-gradient">Assessment</span>'
                '</div>', unsafe_allow_html=True)
    st.caption("Simultaneously evaluates credit eligibility classification and continuous repayment limit in a unified underwriting run.")

    with st.form("joint_form"):
        st.markdown("#### 1. Applicant Profile & Financial Position")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            age = st.number_input("Age", 18, 85, 32, 1, key="j_age")
            salary = st.number_input("Monthly Salary (₹)", 1000, 2000000, 75000, 5000, key="j_sal")
            gender = st.selectbox("Gender", ["Male", "Female"], key="j_gen")
        with c2:
            marital = st.selectbox("Marital Status", ["Married", "Single"], key="j_mar")
            education = st.selectbox("Education", ["Graduate", "Post Graduate", "Professional", "High School"], key="j_edu")
            years_emp = st.number_input("Years Experience", 0.0, 50.0, 4.5, 0.5, key="j_yemp")
        with c3:
            emp_type = st.selectbox("Employment", ["Private", "Government", "Self-employed"], key="j_emp")
            company_type = st.selectbox("Company", ["MNC", "Large Indian", "Startup", "Mid-size", "Small"], key="j_comp")
            dependents = st.number_input("Dependents", 0, 10, 1, 1, key="j_dep")
        with c4:
            house_type = st.selectbox("Residence", ["Rented", "Own", "Family"], key="j_house")
            rent = st.number_input("Monthly Rent (₹)", 0, 300000, 14000, 1000, key="j_rent")
            credit_score = st.number_input("Credit Score", 0, 900, 740, 10, key="j_score")

        st.markdown("#### 2. Monthly Outflows & Balances")
        e1, e2, e3, e4 = st.columns(4)
        with e1:
            groceries = st.number_input("Groceries/Utilities (₹)", 500, 200000, 12000, 1000, key="j_groc")
            travel = st.number_input("Travel/Fuel (₹)", 0, 100000, 4500, 500, key="j_trav")
        with e2:
            school_fees = st.number_input("School Fees (₹)", 0, 100000, 3500, 500, key="j_sch")
            college_fees = st.number_input("College Fees (₹)", 0, 200000, 0, 1000, key="j_col")
        with e3:
            other_exp = st.number_input("Other Expenses (₹)", 0, 200000, 4500, 500, key="j_oth")
            existing_loans = st.selectbox("Existing Loans?", ["No", "Yes"], key="j_exl")
        with e4:
            current_emi = st.number_input("Current EMI Amount (₹)", 0, 500000, 0, 1000, key="j_cemi")
            bank_bal = st.number_input("Bank Savings (₹)", 0, 10000000, 220000, 10000, key="j_bbal")
            emerg_fund = st.number_input("Emergency Fund (₹)", 0, 5000000, 80000, 5000, key="j_efund")

        st.markdown("#### 3. Loan Application Details")
        l1, l2, l3 = st.columns(3)
        with l1:
            req_amount = st.number_input("Requested Loan Amount (₹)", 5000, 5000000, 250000, 10000, key="j_lamt")
        with l2:
            req_tenure = st.number_input("Requested Tenure (Months)", 3, 84, 24, 3, key="j_lten")
        with l3:
            scenario_opts = ["Personal Loan EMI", "Vehicle EMI", "Education EMI", "Home Appliances EMI", "E-commerce Shopping EMI"]
            emi_scenario = st.selectbox("Loan Scenario", scenario_opts, key="j_scen")

        submit_btn = st.form_submit_button("⚡ Run Comprehensive Dual Underwriting Assessment", use_container_width=True, type="primary")

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
            st.error("❌ Submission blocked due to input errors:")
            for err in errors:
                st.error(f"• {err}")
            return

        input_df = pd.DataFrame([{col: applicant_payload[col] for col in FEATURE_NAMES_APPLICATION}])

        with st.spinner("Executing Dual Multi-Task Ensembles..."):
            # 1. Classification
            encoded_clf = clf_pipeline.predict(input_df)[0]
            probs_clf = clf_pipeline.predict_proba(input_df)[0]
            pred_class = le.inverse_transform([encoded_clf])[0]
            prob_dict = {c_name: float(p) for c_name, p in zip(le.classes_, probs_clf)}

            # 2. Regression
            raw_reg = float(reg_pipeline.predict(input_df)[0])
            policy_reg = max(FLOOR_POLICY_LIMIT, raw_reg)

        ratios = calculate_financial_ratios(applicant_payload)
        capacity_margin = policy_reg - ratios["implied_installment"]
        debt_ratio = ratios["implied_installment"] / policy_reg if policy_reg > 0 else 999.0

        rec_id = save_prediction(
            assessment_type="Joint",
            applicant_data=applicant_payload,
            ratios=ratios,
            eligibility_result=pred_class,
            probabilities=prob_dict,
            max_emi_raw=raw_reg,
            max_emi_policy=policy_reg,
            capacity_margin=capacity_margin,
            notes=f"Unified assessment: Class={pred_class}, Capacity=₹{policy_reg:,.2f}, DebtRatio={debt_ratio:.2f}"
        )

        st.markdown("---")
        st.markdown("### 🔍 Multi-Task Decision Matrix")

        res_c1, res_c2, res_c3 = st.columns(3)
        with res_c1:
            st.markdown(f"**Eligibility Classification:**")
            st.markdown(status_badge_html(pred_class), unsafe_allow_html=True)
            st.caption(f"Audit Reference: <code>#{rec_id}</code>", unsafe_allow_html=True)
            if pred_class == "High_Risk":
                high_risk_warning_callout()

        with res_c2:
            st.markdown(metric_card_html("Policy Monthly EMI Limit", f"₹{policy_reg:,.2f}", f"Raw prediction: ₹{raw_reg:,.2f}"), unsafe_allow_html=True)
            if raw_reg < FLOOR_POLICY_LIMIT:
                floor_policy_callout(raw_reg, policy_reg)

        with res_c3:
            st.markdown(metric_card_html(
                "Debt-to-Capacity Ratio",
                f"{debt_ratio:.2f}x",
                f"Installment (₹{ratios['implied_installment']:,.0f}) / Capacity (P/N; 0% APR)"
            ), unsafe_allow_html=True)

        st.markdown("#### Probability Distribution & Coherence")
        render_probability_bars(prob_dict)

        # Cross-Model Consistency Check
        st.markdown("#### ⚖️ Cross-Model Coherence Check")
        if debt_ratio > 1.0 and pred_class == "Not_Eligible":
            st.success("✓ **Models Coherent:** Both models agree that the requested installment exceeds applicant debt capacity.")
        elif debt_ratio <= 0.60 and pred_class == "Eligible":
            st.success("✓ **Models Coherent:** Both models agree that applicant has comfortable capacity surplus.")
        elif 0.70 < debt_ratio <= 1.05 and pred_class == "High_Risk":
            st.warning("⚠️ **Borderline Alignment:** Implied installment is near capacity boundary, aligning with the High_Risk classification.")
        else:
            st.info(f"ℹ️ **Discrepancy Note:** Debt ratio is {debt_ratio:.2f}x while class is '{pred_class}'. Manual underwriter review advised.")
