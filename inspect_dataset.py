import pandas as pd

path = "data/raw/emi_prediction_dataset.csv"
df = pd.read_csv(path)

print("\n--- DATASET SHAPE ---")
print(df.shape)

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())

print("\n--- TARGET COLUMNS ---")
for col in ["emi_eligibility", "max_monthly_emi"]:
    if col in df.columns:
        print(f"\n{col}:")
        if col == "emi_eligibility":
            print(df[col].value_counts(dropna=False))
        else:
            print(df[col].describe())
