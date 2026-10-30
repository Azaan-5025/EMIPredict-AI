import pandas as pd

path = input("Raw CSV path: ").strip()
columns = [c.strip() for c in input("Columns (comma separated): ").split(",")]

df = pd.read_csv(path, dtype=str, keep_default_na=False)
pattern = r"^(-?\d+(?:\.\d+)?)\.0+$"

for col in columns:
    s = df[col]
    numeric = pd.to_numeric(s.str.replace(pattern, r"\1", regex=True), errors="coerce")
    unexplained = s[numeric.isna() & ~s.str.lower().isin(["", "nan", "nan.0"])]
    print(f"\n--- {col}: UNEXPLAINED INVALID VALUES ({len(unexplained)}) ---")
    print("Row index  Value")
    print(unexplained.head(50).to_string())
