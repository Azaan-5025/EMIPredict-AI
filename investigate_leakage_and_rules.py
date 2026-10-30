"""
Phase 6: Deep Investigation of Label Generation Rules & Target Leakage
EMIPredict AI
Audits:
1. Exact formula/logic behind max_monthly_emi
2. Exact decision rules separating Eligible, High_Risk, and Not_Eligible
3. Why 14,545 applicants with max_monthly_emi >= requested_installment are Not_Eligible
4. Relationship between requested_amount, requested_tenure, and targets
5. Defensible feature sets for Classification and Regression
"""

import pandas as pd
import numpy as np

def analyze():
    print("Loading data...")
    df = pd.read_csv("data/processed/emi_cleaned.csv", low_memory=False)
    print(f"Rows: {len(df):,}\n")

    # 1. Financial totals
    expense_cols = [
        "school_fees", "college_fees", "travel_expenses",
        "groceries_utilities", "other_monthly_expenses"
    ]
    rent = df["monthly_rent"].fillna(0)
    total_expenses = df[expense_cols].sum(axis=1) + rent
    disposable_income = df["monthly_salary"] - total_expenses - df["current_emi_amount"]
    df["total_expenses"] = total_expenses
    df["disposable_income"] = disposable_income
    df["req_installment"] = df["requested_amount"] / df["requested_tenure"]

    print("=" * 70)
    print("1. INVESTIGATING max_monthly_emi GENERATION RULE")
    print("=" * 70)
    # Check max_monthly_emi vs disposable_income
    pos_disp = df[df["disposable_income"] > 0].copy()
    pos_disp["ratio_emi_disp"] = pos_disp["max_monthly_emi"] / pos_disp["disposable_income"]
    print("Ratio: max_monthly_emi / disposable_income (where disposable_income > 0):")
    print(pos_disp["ratio_emi_disp"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))

    # Check max_monthly_emi vs monthly_salary
    df["ratio_emi_salary"] = df["max_monthly_emi"] / df["monthly_salary"]
    print("\nRatio: max_monthly_emi / monthly_salary:")
    print(df["ratio_emi_salary"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))

    # Check rows where max_monthly_emi == 500
    at_floor = df[df["max_monthly_emi"] == 500]
    print(f"\nRows at max_monthly_emi == 500: {len(at_floor):,}")
    print(f"  Disposable income <= 0: {(at_floor['disposable_income'] <= 0).sum():,} ({(at_floor['disposable_income'] <= 0).mean()*100:.2f}%)")
    print(f"  Disposable income > 0: {(at_floor['disposable_income'] > 0).sum():,}")
    print(f"  Disposable income range for at_floor rows: [{at_floor['disposable_income'].min()}, {at_floor['disposable_income'].max()}]")

    print("\n" + "=" * 70)
    print("2. INVESTIGATING emi_eligibility GENERATION RULES")
    print("=" * 70)
    # Check condition: can afford requested installment
    df["can_afford"] = df["max_monthly_emi"] >= df["req_installment"]
    print("Crosstab: can_afford (max_monthly_emi >= req_installment) vs emi_eligibility:")
    print(pd.crosstab(df["can_afford"], df["emi_eligibility"], margins=True))

    # Examine the 14,545 anomaly cases: can_afford == True but Not_Eligible
    anomalies = df[df["can_afford"] & (df["emi_eligibility"] == "Not_Eligible")].copy()
    print(f"\nAnalyzing the {len(anomalies):,} rows where max_monthly_emi >= req_installment but Not_Eligible:")
    print("Credit score summary:")
    print(anomalies["credit_score"].describe())
    print(f"  Credit score < 600: {(anomalies['credit_score'] < 600).sum():,} ({(anomalies['credit_score'] < 600).mean()*100:.2f}%)")
    print(f"  Credit score == 0: {(anomalies['credit_score'] == 0).sum():,}")
    print(f"  Credit score NaN: {anomalies['credit_score'].isna().sum():,}")
    print(f"  Credit score >= 600: {(anomalies['credit_score'] >= 600).sum():,}")

    # Check other features for those with credit_score >= 600
    good_cs_anomalies = anomalies[anomalies["credit_score"] >= 600]
    print(f"\n  Of those with credit_score >= 600 ({len(good_cs_anomalies):,} rows):")
    print("  Bank balance vs emergency fund vs salary:")
    print(good_cs_anomalies[["monthly_salary", "bank_balance", "emergency_fund", "years_of_employment", "age"]].describe().T[["mean", "min", "50%", "max"]])

    # Compare Eligible vs High_Risk
    print("\n" + "=" * 70)
    print("3. WHAT DISTINGUISHES Eligible FROM High_Risk?")
    print("=" * 70)
    elig = df[df["emi_eligibility"] == "Eligible"]
    hr = df[df["emi_eligibility"] == "High_Risk"]

    print("Credit score comparison:")
    print("Eligible:")
    print(elig["credit_score"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]))
    print("High_Risk:")
    print(hr["credit_score"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]))

    print("\nFOIR / DTI comparison (current_emi + req_installment) / monthly_salary:")
    df["projected_foir"] = (df["current_emi_amount"] + df["req_installment"]) / df["monthly_salary"]
    print(df.groupby("emi_eligibility")["projected_foir"].describe(percentiles=[0.25, 0.5, 0.75, 0.9]))

    print("\nRatio of req_installment to max_monthly_emi:")
    df["installment_to_max_emi"] = df["req_installment"] / df["max_monthly_emi"]
    print(df.groupby("emi_eligibility")["installment_to_max_emi"].describe(percentiles=[0.25, 0.5, 0.75, 0.9]))

    print("\n" + "=" * 70)
    print("4. CORRELATIONS WITH TARGETS")
    print("=" * 70)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # map emi_eligibility to numeric for correlation check
    df["elig_code"] = df["emi_eligibility"].map({"Not_Eligible": 0, "High_Risk": 1, "Eligible": 2})
    corrs_elig = df[num_cols + ["elig_code"]].corr()["elig_code"].sort_values(ascending=False)
    corrs_emi = df[num_cols + ["elig_code"]].corr()["max_monthly_emi"].sort_values(ascending=False)

    print("\nTop correlations with emi_eligibility:")
    print(corrs_elig.to_string())

    print("\nTop correlations with max_monthly_emi:")
    print(corrs_emi.to_string())

if __name__ == "__main__":
    analyze()
