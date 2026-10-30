"""
Phase 4: Relationship, Data Leakage, and Anomaly Audit Script
EMIPredict AI - Intelligent Financial Risk Assessment Platform
Read-only script to audit:
1. Feature relationships & collinearity (family_size vs dependents, age vs employment)
2. Logical consistency (existing_loans vs current_emi, expenses vs salary)
3. Missing value patterns & row overlap
4. Credit score anomalies (0, >850) and customer profiles
5. Target analysis & leakage checks (max_monthly_emi vs emi_eligibility, requested loan vs targets)
6. EMI scenario distributions
"""

import pandas as pd
import numpy as np

CLEANED_PATH = "data/processed/emi_cleaned.csv"

def run_audit():
    print(f"Loading cleaned dataset from {CLEANED_PATH}...")
    df = pd.read_csv(CLEANED_PATH, low_memory=False)
    print(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns.\n")

    print("=" * 80)
    print("AUDIT 1: HYPOTHESIS TESTING - COLLINEARITY & LOGICAL CHECKS")
    print("=" * 80)

    # 1.1 dependents vs family_size
    diff_fam = df["family_size"] - df["dependents"]
    is_exact_offset = (diff_fam == 1).all()
    print(f"\n[1.1] family_size vs dependents:")
    print(f"  Is family_size - dependents == 1 on EVERY row? {is_exact_offset}")
    print(f"  Distribution of (family_size - dependents):")
    print(diff_fam.value_counts(dropna=False).to_string())

    # 1.2 years_of_employment vs age
    implied_start_age = df["age"] - df["years_of_employment"]
    underage_work = (implied_start_age < 18).sum()
    negative_start = (implied_start_age < 0).sum()
    max_emp_gt_age = (df["years_of_employment"] > df["age"]).sum()
    print(f"\n[1.2] years_of_employment vs age:")
    print(f"  Employment years > Age: {max_emp_gt_age} rows")
    print(f"  Implied starting age < 18: {underage_work} rows ({(underage_work/len(df))*100:.2f}%)")
    print(f"  Implied starting age < 0 (impossible): {negative_start} rows")
    print(f"  Implied starting age summary:\n{implied_start_age.describe().to_string()}")

    # 1.3 existing_loans vs current_emi_amount
    print(f"\n[1.3] existing_loans vs current_emi_amount:")
    loans_no_emi_gt_0 = ((df["existing_loans"] == "No") & (df["current_emi_amount"] > 0)).sum()
    loans_yes_emi_0 = ((df["existing_loans"] == "Yes") & (df["current_emi_amount"] == 0)).sum()
    print(f"  existing_loans == 'No' but current_emi_amount > 0: {loans_no_emi_gt_0} rows")
    print(f"  existing_loans == 'Yes' but current_emi_amount == 0: {loans_yes_emi_0} rows")
    print("  current_emi_amount summary grouped by existing_loans:")
    print(df.groupby("existing_loans")["current_emi_amount"].describe().to_string())

    # 1.4 monthly_rent vs house_type
    print(f"\n[1.4] monthly_rent missingness vs house_type:")
    print(pd.crosstab(df["house_type"], df["monthly_rent"].isna(), rownames=["house_type"], colnames=["rent_is_na"], margins=True).to_string())
    print("\n  monthly_rent summary by house_type (non-missing):")
    print(df.groupby("house_type")["monthly_rent"].describe().to_string())

    print("\n" + "=" * 80)
    print("AUDIT 2: MISSING VALUE ROW-OVERLAP ANALYSIS")
    print("=" * 80)
    missing_cols = [c for c in df.columns if df[c].isna().sum() > 0]
    print(f"Columns with missing values: {missing_cols}")
    missing_mask = df[missing_cols].isna()
    print(f"\nMissing count per column:")
    print(missing_mask.sum().to_string())

    # Pattern of missingness
    patterns = missing_mask.groupby(missing_cols).size().reset_index(name="row_count")
    print(f"\nMissing value combination patterns:")
    print(patterns.to_string(index=False))

    all_missing_rows = missing_mask.any(axis=1).sum()
    print(f"\nTotal rows with at least one missing value: {all_missing_rows:,} ({(all_missing_rows/len(df))*100:.2f}%)")

    print("\n" + "=" * 80)
    print("AUDIT 3: CREDIT SCORE ANOMALIES & BEHAVIOR")
    print("=" * 80)
    cs = df["credit_score"]
    cs_zero = df[cs == 0]
    cs_high = df[cs > 850]
    cs_normal = df[cs.between(300, 850)]
    cs_na = df[cs.isna()]

    print(f"Credit Score Group Counts:")
    print(f"  Zero (0): {len(cs_zero):,} rows")
    print(f"  Normal (300-850): {len(cs_normal):,} rows")
    print(f"  Above 850 (>850 to {cs.max()}): {len(cs_high):,} rows")
    print(f"  Missing (NaN): {len(cs_na):,} rows")

    print("\nEligibility breakdown by Credit Score Group:")
    cs_group = pd.cut(cs, bins=[-np.inf, -0.1, 0.5, 299, 850, np.inf], labels=["<0", "Zero", "1-299", "300-850", ">850"])
    cs_group = cs_group.cat.add_categories(["Missing"]).fillna("Missing")
    print(pd.crosstab(cs_group, df["emi_eligibility"], normalize="index").apply(lambda s: s.map(lambda v: f"{v*100:.2f}%")).to_string())
    print("\nCounts:")
    print(pd.crosstab(cs_group, df["emi_eligibility"], margins=True).to_string())

    print("\nExisting loans breakdown for Credit Score == 0:")
    print(cs_zero["existing_loans"].value_counts(normalize=True).map(lambda v: f"{v*100:.2f}%").to_string())

    print("\n" + "=" * 80)
    print("AUDIT 4: TARGET RELATIONSHIPS & LEAKAGE INVESTIGATION")
    print("=" * 80)

    # 4.1 Relationship between emi_eligibility and max_monthly_emi
    print("\n[4.1] max_monthly_emi summary by emi_eligibility:")
    agg_target = df.groupby("emi_eligibility")["max_monthly_emi"].agg(
        count="count",
        min="min",
        p25=lambda x: x.quantile(0.25),
        median="median",
        mean="mean",
        p75=lambda x: x.quantile(0.75),
        max="max",
        eq_500=lambda x: (x == 500).sum(),
        pct_500=lambda x: f"{(x == 500).mean()*100:.2f}%",
        gt_50k=lambda x: (x > 50000).sum()
    )
    print(agg_target.to_string())

    # Check Not_Eligible vs 500
    not_elig = df[df["emi_eligibility"] == "Not_Eligible"]
    not_elig_non_500 = (not_elig["max_monthly_emi"] != 500).sum()
    print(f"\nNot_Eligible rows with max_monthly_emi != 500: {not_elig_non_500}")
    if not_elig_non_500 > 0:
        print("Distribution of max_monthly_emi for Not_Eligible:")
        print(not_elig["max_monthly_emi"].describe().to_string())

    # Check Eligible & High_Risk with max_monthly_emi == 500
    elig_500 = (df[df["emi_eligibility"] == "Eligible"]["max_monthly_emi"] == 500).sum()
    high_risk_500 = (df[df["emi_eligibility"] == "High_Risk"]["max_monthly_emi"] == 500).sum()
    print(f"Eligible rows with max_monthly_emi == 500: {elig_500}")
    print(f"High_Risk rows with max_monthly_emi == 500: {high_risk_500}")

    # 4.2 Can max_monthly_emi perfectly classify emi_eligibility?
    print(f"\n[4.2] Separation check: max_monthly_emi overlap across classes:")
    print(f"  Not_Eligible range: [{not_elig['max_monthly_emi'].min()}, {not_elig['max_monthly_emi'].max()}]")
    elig = df[df["emi_eligibility"] == "Eligible"]
    print(f"  Eligible range: [{elig['max_monthly_emi'].min()}, {elig['max_monthly_emi'].max()}]")
    hr = df[df["emi_eligibility"] == "High_Risk"]
    print(f"  High_Risk range: [{hr['max_monthly_emi'].min()}, {hr['max_monthly_emi'].max()}]")

    # 4.3 Requested Loan vs Targets
    # Standard monthly installment calculation check: does requested_amount / requested_tenure match max_monthly_emi?
    simple_installment = df["requested_amount"] / df["requested_tenure"]
    ratio = df["max_monthly_emi"] / simple_installment
    print(f"\n[4.3] requested_amount / requested_tenure vs max_monthly_emi:")
    print(f"  Correlation(requested_amount, max_monthly_emi): {df['requested_amount'].corr(df['max_monthly_emi']):.4f}")
    print(f"  Correlation(requested_tenure, max_monthly_emi): {df['requested_tenure'].corr(df['max_monthly_emi']):.4f}")
    print(f"  Correlation(simple_installment, max_monthly_emi): {simple_installment.corr(df['max_monthly_emi']):.4f}")
    print(f"  Ratio (max_monthly_emi / (req_amt / req_tenure)) summary:")
    print(ratio.describe().to_string())

    # Is max_monthly_emi >= simple_installment when Eligible?
    can_afford_requested = df["max_monthly_emi"] >= simple_installment
    print(f"\n  Does (max_monthly_emi >= requested_amount / requested_tenure) match emi_eligibility?")
    print(pd.crosstab(df["emi_eligibility"], can_afford_requested, rownames=["emi_eligibility"], colnames=["max_emi >= req_installment"]).to_string())

    # 4.4 Financial Capacity / FOIR formula check
    # Check disposable income = monthly_salary - total_expenses - current_emi_amount
    expense_cols = [
        "school_fees", "college_fees", "travel_expenses",
        "groceries_utilities", "other_monthly_expenses"
    ]
    total_expenses = df[expense_cols].sum(axis=1) + df["monthly_rent"].fillna(0)
    disposable_income = df["monthly_salary"] - total_expenses - df["current_emi_amount"]

    print(f"\n[4.4] Financial Capacity Analysis:")
    print(f"  Negative disposable income rows: {(disposable_income < 0).sum()} ({(disposable_income < 0).mean()*100:.2f}%)")
    print(f"  Correlation(monthly_salary, max_monthly_emi): {df['monthly_salary'].corr(df['max_monthly_emi']):.4f}")
    print(f"  Correlation(disposable_income, max_monthly_emi): {disposable_income.corr(df['max_monthly_emi']):.4f}")

    emi_to_salary = df["max_monthly_emi"] / df["monthly_salary"]
    print(f"\n  max_monthly_emi / monthly_salary (FOIR proxy) summary:")
    print(emi_to_salary.describe().to_string())
    print("  FOIR proxy by eligibility:")
    print(df.groupby("emi_eligibility").apply(lambda g: (g["max_monthly_emi"] / g["monthly_salary"]).describe()).unstack().to_string())

    print("\n" + "=" * 80)
    print("AUDIT 5: HIGH-VALUE MAX EMI (> 50,000) INVESTIGATION")
    print("=" * 80)
    high_emi = df[df["max_monthly_emi"] > 50000]
    print(f"Count of max_monthly_emi > 50,000: {len(high_emi)} rows")
    print(f"  Eligibility distribution:\n{high_emi['emi_eligibility'].value_counts().to_string()}")
    print(f"  Monthly salary summary in high EMI group:\n{high_emi['monthly_salary'].describe().to_string()}")
    print(f"  Max EMI / Monthly salary ratio in high EMI group:\n{(high_emi['max_monthly_emi'] / high_emi['monthly_salary']).describe().to_string()}")
    print(f"  Credit score summary in high EMI group:\n{high_emi['credit_score'].describe().to_string()}")

    print("\n" + "=" * 80)
    print("AUDIT 6: EMI SCENARIO DISTRIBUTION & RELATIONSHIPS")
    print("=" * 80)
    print(pd.crosstab(df["emi_scenario"], df["emi_eligibility"], margins=True).to_string())
    print("\nmax_monthly_emi by emi_scenario:")
    print(df.groupby("emi_scenario")["max_monthly_emi"].describe().to_string())

    print("\n" + "=" * 80)
    print("AUDIT 7: DATA LEAKAGE CONCLUSIONS & RECOMMENDATIONS")
    print("=" * 80)
    print("Summary of leakage risks:")
    print("1. max_monthly_emi vs emi_eligibility:")
    print("   If max_monthly_emi is used as an input to classification, it leaks the regression target into classification.")
    print("   If emi_eligibility is used as an input to regression, it leaks the classification target into regression.")
    print("   CONCLUSION: Neither target can be used as a feature in the other's model.")
    print("2. family_size vs dependents:")
    if is_exact_offset:
        print("   family_size - dependents is exactly 1 on all rows! Perfect multicollinearity.")
        print("   RECOMMENDATION: Drop one or use only one in linear models / risk scoring.")
    else:
        print("   family_size and dependents are not perfectly collinear.")
    print("\n--- PHASE 4 AUDIT COMPLETE ---")

if __name__ == "__main__":
    run_audit()
