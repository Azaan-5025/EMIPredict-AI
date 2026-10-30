"""
Data splitting module for EMIPredict AI.
Performs reproducible, stratified train/test split (80/20) with random_state=42.
Ensures identical class ratios across train and test sets and preserves raw data.
"""

from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

PROCESSED_DATA_PATH = Path("data/processed/emi_cleaned.csv")
TRAIN_DATA_PATH = Path("data/processed/train.csv")
TEST_DATA_PATH = Path("data/processed/test.csv")
RANDOM_STATE = 42
TEST_SIZE = 0.2

def split_dataset(
    input_path: Path = PROCESSED_DATA_PATH,
    output_dir: Path = Path("data/processed"),
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs a reproducible stratified train/test split on emi_eligibility.
    Saves train.csv and test.csv in output_dir.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading cleaned dataset from {input_path}...")
    df = pd.read_csv(input_path, low_memory=False)
    total_rows = len(df)
    print(f"Loaded {total_rows:,} rows across {len(df.columns)} columns.")

    # Stratified split on emi_eligibility
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["emi_eligibility"],
        shuffle=True
    )

    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"

    print(f"Saving train split ({len(train_df):,} rows) to {train_path}...")
    train_df.to_csv(train_path, index=False)

    print(f"Saving test split ({len(test_df):,} rows) to {test_path}...")
    test_df.to_csv(test_path, index=False)

    # Validate stratification
    print("\n--- STRATIFICATION VERIFICATION ---")
    train_dist = train_df["emi_eligibility"].value_counts(normalize=True)
    test_dist = test_df["emi_eligibility"].value_counts(normalize=True)
    overall_dist = df["emi_eligibility"].value_counts(normalize=True)

    summary = pd.DataFrame({
        "Overall (%)": (overall_dist * 100).round(2),
        "Train (%)": (train_dist * 100).round(2),
        "Test (%)": (test_dist * 100).round(2)
    })
    print(summary.to_string())

    return train_df, test_df

def load_train_test_data(
    data_dir: Path = Path("data/processed")
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads train.csv and test.csv from data_dir.
    If they do not exist, runs split_dataset first.
    """
    data_dir = Path(data_dir)
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    if not train_path.exists() or not test_path.exists():
        print("Train/test splits not found. Creating splits...")
        return split_dataset(output_dir=data_dir)

    train_df = pd.read_csv(train_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)
    return train_df, test_df

if __name__ == "__main__":
    split_dataset()
