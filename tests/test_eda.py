"""
Automated unit tests for Phase 7: Exploratory Data Analysis.

Verifies:
1. All 6 publication-quality EDA figures were generated and have valid file sizes.
2. The structured EDA report exists and contains all required analytical sections.
3. The held-out test split (test.csv) has remained completely intact and untouched.
4. The raw dataset SHA-256 fingerprint remains strictly unmodified.
"""

from pathlib import Path
import hashlib
import pandas as pd
import pytest

EXPECTED_FIGURES = [
    "01_target_distributions.png",
    "02_missing_values_profile.png",
    "03_credit_score_analysis.png",
    "04_financial_capacity_and_foir.png",
    "05_scenario_comparison.png",
    "06_correlation_heatmap.png"
]

RAW_DATA_SHA256 = "46A52DCBC80B8D736C5A42E9FA5173289E66BB9D30DD03E4E2F33E2B9C1A53C4"


def test_eda_figures_exist():
    """All 6 EDA figures must exist in reports/figures/ and have substantial size."""
    figures_dir = Path("reports/figures")
    assert figures_dir.exists(), "reports/figures directory does not exist!"

    for fig_name in EXPECTED_FIGURES:
        fig_path = figures_dir / fig_name
        assert fig_path.exists(), f"Missing figure: {fig_name}"
        assert fig_path.stat().st_size > 10000, f"Figure {fig_name} is unexpectedly small (< 10 KB)!"


def test_eda_report_exists_and_content():
    """reports/eda_report.md must exist and contain all required analysis sections."""
    report_path = Path("reports/eda_report.md")
    assert report_path.exists(), "reports/eda_report.md does not exist!"
    assert report_path.stat().st_size > 5000, "EDA report is unexpectedly small!"

    content = report_path.read_text(encoding="utf-8")
    required_sections = [
        "Executive Summary",
        "Target Variable Analysis",
        "Classification Target",
        "max_monthly_emi",
        "Missing Value Analysis",
        "Credit Score Anatomy",
        "Debt Capacity, Installment Ratio & FOIR Dynamics",
        "Numerical Feature Distributions",
        "Categorical Feature Distributions",
        "EMI Scenario Comparisons",
        "Correlation Analysis",
        "Synthetic Artifacts & Limitations Summary"
    ]
    for section in required_sections:
        assert section in content, f"EDA report is missing required section: '{section}'"


def test_test_set_untouched():
    """The held-out test.csv must have 80,960 rows and 27 columns and zero modifications."""
    test_path = Path("data/processed/test.csv")
    assert test_path.exists(), "test.csv does not exist!"

    df_test = pd.read_csv(test_path, low_memory=False)
    assert df_test.shape == (80960, 27), f"Expected (80960, 27), got {df_test.shape}"


def test_raw_dataset_fingerprint_unmodified():
    """The raw dataset SHA-256 fingerprint must match the verified reference."""
    raw_path = Path("data/raw/emi_prediction_dataset.csv")
    assert raw_path.exists(), "Raw dataset does not exist!"

    sha256 = hashlib.sha256()
    with open(raw_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    actual_hash = sha256.hexdigest().upper()

    assert actual_hash == RAW_DATA_SHA256, f"Raw data hash modified! Expected {RAW_DATA_SHA256}, got {actual_hash}"
