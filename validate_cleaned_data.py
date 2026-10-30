import pandas as pd

df = pd.read_csv(
    "data/processed/emi_cleaned.csv",
    low_memory=False
)

print("\n--- REMAINING INVALID SALARIES ---")
print(df.loc[df["monthly_salary"].isna(), "monthly_salary"].to_string())

print("\n--- REMAINING MISSING VALUES ---")
print(df.isna().sum()[df.isna().sum() > 0].to_string())

print("\n--- CREDIT SCORE DISTRIBUTION ---")
credit = df["credit_score"]
print("Below 300:", int((credit < 300).sum()))
print("300 to 850:", int(credit.between(300, 850).sum()))
print("Above 850:", int((credit > 850).sum()))
print("Missing:", int(credit.isna().sum()))

print("\n--- MAXIMUM EMI TARGET ---")
target = df["max_monthly_emi"]
print(target.describe().to_string())
print("Exactly 500:", int((target == 500).sum()))
print("Above 50000:", int((target > 50000).sum()))

print("\n--- VALIDATION COMPLETE ---")
