import pandas as pd

df = pd.read_csv("data/raw/emi_prediction_dataset.csv", low_memory=False)

for col in ["age", "monthly_salary", "bank_balance"]:
    converted = pd.to_numeric(df[col], errors="coerce")
    invalid = df.loc[converted.isna(), col]

    print(f"\n--- {col}: INVALID VALUE SUMMARY ---")
    print("Invalid count:", int(converted.isna().sum()))
    print("Distinct invalid values:", int(invalid.nunique()))
    print("Most common invalid values:")
    print(invalid.value_counts(dropna=False).head(10).to_string())

print("\n--- CREDIT SCORE INVESTIGATION ---")
credit = df["credit_score"]
print("Missing:", int(credit.isna().sum()))
print("Zero:", int((credit == 0).sum()))
print("Below 300:", int((credit < 300).sum()))
print("Above 850:", int((credit > 850).sum()))

print("\n--- REGRESSION TARGET ABOVE SPECIFIED LIMIT ---")
target = df["max_monthly_emi"]
print("Above 50000:", int((target > 50000).sum()))
print("At or below 500:", int((target <= 500).sum()))

print("\n--- RENT AND HOUSE TYPE ---")
print(pd.crosstab(
    df["house_type"],
    df["monthly_rent"].isna(),
    margins=True
).to_string())

print("\n--- INVESTIGATION COMPLETE ---")
