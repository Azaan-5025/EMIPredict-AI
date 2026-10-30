"""
EMI Eligibility Prediction View: Multi-Class Financial Risk Classifier.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
import streamlit as st

from app.components import (
    status_badge_html,
    render_probability_bars,
    high_risk_warning_callout
)
from app.validation import validate_applicant_inputs, calculate_financial_ratios
from app.db import save_prediction
from src.features.pipeline import FEATURE_NAMES_APPLICATION


def render_eligibility_view(clf_pipeline, le):
    """Renders the applicant eligibility underwriting interface."""
    st.markdown('<div class="brand-title" style="font-size: 2rem; font-weight: 700;">'
                'EMI Eligibility <span class="brand-gradient">Assessment</span>'
                '</div>', unsafe_allow_html=True)
    st.caption("Evaluates applicant repayment capacity across three risk tiers: Eligible, High Risk, or Not Eligible.")

    # Preset Profiles for Rapid Testing
    preset_choice = st.selectbox(
        "⚡ Quick-Fill Preset Applicant Profiles (Optional):",
        ["Custom Manual Entry", "Profile A: Prime Salaried Borrower", "Profile B: Borderline Moderate Risk", "Profile C: Over-Leveraged Insolvent"],
        index=0
    )

    # Defaults setup based on preset
    defaults = {
        "gender": "Male", "marital_status": "Married", "education": "Graduate",
        "employment_type": "Private", "company_type": "MNC", "house_type": "Rented",
        "existing_loans": "No", "emi_scenario": "Personal Loan EMI",
        "age": 34, "salary": 65000, "years_emp": 5.0, "rent": 12000,
        "dependents": 1, "school_fees": 3000, "college_fees": 0,
        "travel": 4500, "groceries": 11000, "other_exp": 5000,
        "current_emi": 0, "credit_score": 750, "bank_bal": 160000,
        "emerg_fund": 70000, "req_amount": 150000, "req_tenure": 24
    }

    if preset_choice == "Profile A: Prime Salaried Borrower":
        defaults.update({
            "salary": 110000, "rent": 15000, "dependents": 1, "school_fees": 4000,
            "groceries": 14000, "other_exp": 6000, "credit_score": 790,
            "bank_bal": 350000, "emerg_fund": 150000, "req_amount": 120000, "req_tenure": 24
        })
    elif preset_choice == "Profile B: Borderline Moderate Risk":
        defaults.update({
            "salary": 50000, "rent": 10000, "dependents": 2, "school_fees": 3000,
            "travel": 3500, "groceries": 11000, "other_exp": 4000, "existing_loans": "No",
            "current_emi": 0, "credit_score": 670, "bank_bal": 75000,
            "emerg_fund": 30000, "req_amount": 160000, "req_tenure": 24
        })
    elif preset_choice == "Profile C: Over-Leveraged Insolvent":
        defaults.update({
            "salary": 32000, "rent": 12000, "dependents": 3, "school_fees": 5000,
            "groceries": 12000, "other_exp": 5000, "existing_loans": "Yes",
            "current_emi": 14000, "credit_score": 580, "bank_bal": 15000,
            "emerg_fund": 5000, "req_amount": 350000, "req_tenure": 18
        })

    with st.form("eligibility_form"):
        # Section 1: Demographics & Professional Profile
        st.markdown("#### 1. Demographic & Professional Profile")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            age = st.number_input("Age (Years)", min_value=18, max_value=85, value=int(defaults["age"]), step=1)
            gender = st.selectbox("Gender", ["Male", "Female"], index=0 if defaults["gender"] == "Male" else 1)
        with c2:
            marital = st.selectbox("Marital Status", ["Married", "Single"], index=0 if defaults["marital_status"] == "Married" else 1)
            education = st.selectbox("Education Level", ["Graduate", "Post Graduate", "Professional", "High School"],
                                     index=["Graduate", "Post Graduate", "Professional", "High School"].index(defaults["education"]))
        with c3:
            emp_type = st.selectbox("Employment Sector", ["Private", "Government", "Self-employed"],
                                    index=["Private", "Government", "Self-employed"].index(defaults["employment_type"]))
            company_type = st.selectbox("Company Category", ["Large Indian", "MNC", "Startup", "Mid-size", "Small"],
                                        index=["Large Indian", "MNC", "Startup", "Mid-size", "Small"].index(defaults["company_type"]))
        with c4:
            years_emp = st.number_input("Work Experience (Years)", min_value=0.0, max_value=50.0,
                                        value=float(defaults["years_emp"]), step=0.5)
            dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=int(defaults["dependents"]), step=1)

        # Section 2: Income & Living Commitments
        st.markdown("#### 2. Monthly Income, Housing & Living Obligations")
        inc_c1, inc_c2, inc_c3, inc_c4 = st.columns(4)
        with inc_c1:
            salary = st.number_input("Monthly Salary (₹)", min_value=1000, max_value=2000000, value=int(defaults["salary"]), step=5000)
            house_type = st.selectbox("Residential Status", ["Rented", "Own", "Family"],
                                      index=["Rented", "Own", "Family"].index(defaults["house_type"]))
        with inc_c2:
            rent = st.number_input("Monthly Rent (₹)", min_value=0, max_value=300000, value=int(defaults["rent"]), step=1000)
            school_fees = st.number_input("School Fees (₹)", min_value=0, max_value=100000, value=int(defaults["school_fees"]), step=500)
        with inc_c3:
            college_fees = st.number_input("College Tuition (₹)", min_value=0, max_value=200000, value=int(defaults["college_fees"]), step=1000)
            travel = st.number_input("Commute & Travel (₹)", min_value=0, max_value=100000, value=int(defaults["travel"]), step=500)
        with inc_c4:
            groceries = st.number_input("Groceries & Utilities (₹)", min_value=500, max_value=200000, value=int(defaults["groceries"]), step=1000)
            other_exp = st.number_input("Other Living Expenses (₹)", min_value=0, max_value=200000, value=int(defaults["other_exp"]), step=500)

        # Section 3: Existing Debt & Balance Sheet
        st.markdown("#### 3. Existing Debt & Liquid Assets")
        debt_c1, debt_c2, debt_c3 = st.columns(3)
        with debt_c1:
            existing_loans = st.selectbox("Existing Active Loans?", ["No", "Yes"], index=0 if defaults["existing_loans"] == "No" else 1)
            current_emi = st.number_input("Current Monthly EMI Obligations (₹)", min_value=0, max_value=500000,
                                          value=int(defaults["current_emi"]), step=1000)
        with debt_c2:
            credit_score = st.number_input("Credit Bureau Score (CIBIL/Experian)", min_value=0, max_value=900,
                                           value=int(defaults["credit_score"]), step=10,
                                           help="Enter 0 if applicant has no credit bureau footprint (New-to-Credit).")
        with debt_c3:
            bank_bal = st.number_input("Bank Savings Balance (₹)", min_value=0, max_value=10000000, value=int(defaults["bank_bal"]), step=10000)
            emerg_fund = st.number_input("Liquid Emergency Fund (₹)", min_value=0, max_value=5000000, value=int(defaults["emerg_fund"]), step=5000)

        # Section 4: Proposed Loan Request
        st.markdown("#### 4. Loan Request Details")
        loan_c1, loan_c2, loan_c3 = st.columns(3)
        with loan_c1:
            req_amount = st.number_input("Requested Loan Amount (₹)", min_value=5000, max_value=5000000,
                                         value=int(defaults["req_amount"]), step=10000)
        with loan_c2:
            req_tenure = st.number_input("Requested Loan Tenure (Months)", min_value=3, max_value=84,
                                         value=int(defaults["req_tenure"]), step=3)
        with loan_c3:
            scenario_opts = [
                "Personal Loan EMI", "Vehicle EMI", "Education EMI",
                "Home Appliances EMI", "E-commerce Shopping EMI"
            ]
            emi_scenario = st.selectbox("Loan Purpose / Scenario", scenario_opts,
                                        index=scenario_opts.index(defaults["emi_scenario"]))

        submit_btn = st.form_submit_button("🔍 Run Eligibility Assessment", use_container_width=True, type="primary")

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

        # Run Validation
        is_valid, errors, warnings = validate_applicant_inputs(applicant_payload)
        for warn in warnings:
            st.warning(f"⚠️ {warn}")

        if not is_valid:
            st.error("❌ Submission blocked due to input validation errors:")
            for err in errors:
                st.error(f"• {err}")
            return

        # Prepare DataFrame strictly matching FEATURE_NAMES_APPLICATION schema
        input_df = pd.DataFrame([{col: applicant_payload[col] for col in FEATURE_NAMES_APPLICATION}])

        with st.spinner("Executing XGBoost Balanced Classification Pipeline..."):
            encoded_pred = clf_pipeline.predict(input_df)[0]
            probs = clf_pipeline.predict_proba(input_df)[0]
            predicted_class = le.inverse_transform([encoded_pred])[0]
            prob_dict = {c_name: float(p) for c_name, p in zip(le.classes_, probs)}

        ratios = calculate_financial_ratios(applicant_payload)

        # Save to SQLite History
        rec_id = save_prediction(
            assessment_type="Eligibility",
            applicant_data=applicant_payload,
            ratios=ratios,
            eligibility_result=predicted_class,
            probabilities=prob_dict,
            notes=f"Evaluated via XGBoost Balanced. Implied installment ₹{ratios['implied_installment']:,.2f}"
        )

        st.markdown("---")
        st.markdown("### 📋 Credit Decision & Risk Assessment Result")

        res_c1, res_c2 = st.columns([1.2, 2.0])
        with res_c1:
            st.markdown(f"**Predicted Assessment:**")
            st.markdown(status_badge_html(predicted_class), unsafe_allow_html=True)
            st.markdown(f"<small>Audit Reference ID: <code>#{rec_id}</code></small>", unsafe_allow_html=True)
            st.markdown(f"**Implied Monthly Installment (P/N):** ₹{ratios['implied_installment']:,.2f}")
            st.caption("Installment calculated as P/N assuming 0% nominal interest and zero fees.")
            st.markdown(f"**Projected FOIR:** {ratios['projected_foir']:.1f}%")

        with res_c2:
            st.markdown("**Predicted Class Probabilities:**")
            render_probability_bars(prob_dict)

        # High_Risk Warning
        if predicted_class == "High_Risk":
            high_risk_warning_callout()
        elif predicted_class == "Not_Eligible":
            st.markdown("""
            <div class="alert-box alert-danger">
                <strong>✕ DECLINE INDICATION:</strong> The applicant's implied loan installment substantially exceeds
                estimated debt capacity. Projected total obligations consume a critical portion of monthly earnings.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-box alert-info">
                <strong>✓ FAVORABLE CREDIT PROFILE:</strong> Repayment cash flow comfortably satisfies the requested loan obligations.
            </div>
            """, unsafe_allow_html=True)
