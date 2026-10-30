import pandas as pd

path = "data/raw/emi_prediction_dataset.csv"

df = pd.read_csv(path, low_memory=False)

print("\n--- AGE VALUES THAT CANNOT BE CONVERTED TO NUMBERS ---")
age = pd.to_numeric(df["age"], errors="coerce")
print("Invalid or missing after conversion:", int(age.isna().sum()))
print("Numeric age range:", age.min(), "to", age.max())

print("\n--- MONTHLY SALARY CONVERSION ---")
salary = pd.to_numeric(df["monthly_salary"], errors="coerce")
print("Invalid or missing after conversion:", int(salary.isna().sum()))
print("Numeric salary range:", salary.min(), "to", salary.max())

print("\n--- BANK BALANCE CONVERSION ---")
balance = pd.to_numeric(df["bank_balance"], errors="coerce")
print("Invalid or missing after conversion:", int(balance.isna().sum()))
print("Numeric balance range:", balance.min(), "to", balance.max())

print("\n--- NUMERIC COLUMN RANGES ---")
cols = [
    "years_of_employment", "monthly_rent", "family_size",
    "dependents", "school_fees", "college_fees",
    "travel_expenses", "groceries_utilities",
    "other_monthly_expenses", "current_emi_amount",
    "credit_score", "emergency_fund", "requested_amount",
    "requested_tenure", "max_monthly_emi"
]
print(df[cols].describe().T.to_string())

print("\n--- CATEGORICAL VALUES ---")
for col in [
    "education", "employment_type", "company_type",
    "house_type", "existing_loans", "emi_scenario",
    "emi_eligibility"
]:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False).head(15).to_string())

print("\n--- TARGET RELATIONSHIP CHECK ---")
print("\nMaximum EMI by eligibility:")
print(df.groupby("emi_eligibility")["max_monthly_emi"]
      .agg(["count", "mean", "median", "min", "max"]).to_string())

print("\n--- DATA QUALITY CHECK COMPLETE ---")
