import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/emi_prediction_dataset.csv")
OUTPUT_PATH = Path("data/processed/emi_cleaned.csv")

df = pd.read_csv(RAW_PATH, low_memory=False)
report = []

# Correct only duplicated decimal suffixes such as 18000.0.0.
def repair_numeric_format(series, column):
    original = series.astype("string")
    repaired = original.str.replace(
        r"^(-?\d+(?:\.\d+)?)(?:\.0+)+$",
        r"\1",
        regex=True
    )
    numeric = pd.to_numeric(repaired, errors="coerce")

    report.append({
        "column": column,
        "original_nonmissing": int(original.notna().sum()),
        "numeric_after_repair": int(numeric.notna().sum()),
        "still_missing_or_invalid": int(numeric.isna().sum())
    })
    return numeric

for col in ["age", "monthly_salary", "bank_balance"]:
    df[col] = repair_numeric_format(df[col], col)

print("\n--- NUMERIC REPAIR REPORT ---")
print(pd.DataFrame(report).to_string(index=False))

print("\n--- VALIDATION COUNTS ---")
print("Age outside 18-100:", int((~df["age"].between(18, 100)).sum()))
print("Salary <= 0:", int((df["monthly_salary"] <= 0).sum()))
print("Bank balance missing:", int(df["bank_balance"].isna().sum()))

print("\n--- CREDIT SCORE FLAGS ---")
credit = df["credit_score"]
print("Missing:", int(credit.isna().sum()))
print("Zero:", int((credit == 0).sum()))
print("Below 300:", int((credit < 300).sum()))
print("Above 850:", int((credit > 850).sum()))

print("\n--- REGRESSION TARGET FLAGS ---")
target = df["max_monthly_emi"]
print("Below 500:", int((target < 500).sum()))
print("Equal to 500:", int((target == 500).sum()))
print("Above 50000:", int((target > 50000).sum()))

print("\n--- SAVE CLEANED COPY ---")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)
print("Saved:", OUTPUT_PATH.resolve())
print("Rows:", len(df))
print("Columns:", len(df.columns))
