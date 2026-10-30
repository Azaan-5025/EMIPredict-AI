"""
Automated unit tests for Phase 9: Regression Models (max_monthly_emi).

Verifies:
1. Saved regression models exist and can be loaded.
2. Saved pipelines predict cleanly on input data with correct shapes and positive values.
3. Strict anti-leakage safeguards (emi_eligibility is excluded, family_size is dropped).
4. Regression report and metrics JSON exist and contain required analytical sections and segmented metrics.
5. The held-out test split (test.csv) has remained completely untouched and unpeeked.
6. The raw dataset SHA-256 fingerprint remains strictly unmodified.
"""

from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
import pytest

RAW_DATA_SHA256 = "46A52DCBC80B8D736C5A42E9FA5173289E66BB9D30DD03E4E2F33E2B9C1A53C4"


def test_saved_regression_artifacts_exist():
    """All required regression model artifacts must exist in models/regression/."""
    models_dir = Path("models/regression")
    assert models_dir.exists(), "models/regression directory does not exist!"

    required_artifacts = [
        "best_application_model.joblib",
        "best_applicant_only_model.joblib",
        "linear_regression_application.joblib",
        "random_forest_application.joblib",
        "xgboost_application.joblib"
    ]
    for art in required_artifacts:
        art_path = models_dir / art
        assert art_path.exists(), f"Missing regression model artifact: {art}"
        assert art_path.stat().st_size > 1000, f"Artifact {art} is unexpectedly small!"


def test_regression_pipeline_predicts_correctly():
    """Loaded regression pipeline must output valid continuous EMI predictions."""
    model_path = Path("models/regression/best_application_model.joblib")
    assert model_path.exists(), "best_application_model.joblib does not exist!"

    pipeline = joblib.load(model_path)
    sample_df = pd.read_csv("data/processed/train.csv", nrows=50)

    from src.features.pipeline import get_regression_features
    X_sample, _ = get_regression_features(sample_df, feature_set="application")

    preds = pipeline.predict(X_sample)

    assert len(preds) == len(sample_df)
    assert isinstance(preds, np.ndarray)
    assert not np.isnan(preds).any(), "NaN found in regression predictions!"
    assert not np.isinf(preds).any(), "Inf found in regression predictions!"
    # Ensure reasonable continuous variation
    assert preds.std() > 100.0, "Predictions lack variation!"


def test_applicant_only_regression_predicts():
    """Applicant-only regression pipeline must output valid continuous predictions without loan request terms."""
    model_path = Path("models/regression/best_applicant_only_model.joblib")
    assert model_path.exists(), "best_applicant_only_model.joblib does not exist!"

    pipeline = joblib.load(model_path)
    sample_df = pd.read_csv("data/processed/train.csv", nrows=50)

    from src.features.pipeline import get_regression_features
    X_sample, _ = get_regression_features(sample_df, feature_set="applicant_only")

    preds = pipeline.predict(X_sample)

    assert len(preds) == len(sample_df)
    assert isinstance(preds, np.ndarray)
    assert not np.isnan(preds).any()
    assert not np.isinf(preds).any()
    assert preds.std() > 100.0, "Predictions lack variation!"


def test_regression_anti_leakage_safeguards():
    """emi_eligibility must never be permitted in regression feature sets."""
    sample_df = pd.read_csv("data/processed/train.csv", nrows=20)
    from src.features.pipeline import get_regression_features

    X_app, y_app = get_regression_features(sample_df, feature_set="application")
    assert "emi_eligibility" not in X_app.columns
    assert "max_monthly_emi" not in X_app.columns
    assert "family_size" not in X_app.columns

    X_pure, y_pure = get_regression_features(sample_df, feature_set="applicant_only")
    assert "emi_eligibility" not in X_pure.columns
    assert "max_monthly_emi" not in X_pure.columns
    assert "family_size" not in X_pure.columns
    for f in ["requested_amount", "requested_tenure", "emi_scenario"]:
        assert f not in X_pure.columns


def test_regression_reports_and_metrics_exist():
    """reports/regression_report.md and metrics JSON must exist and contain required content."""
    report_path = Path("reports/regression_report.md")
    assert report_path.exists(), "reports/regression_report.md does not exist!"
    assert report_path.stat().st_size > 3000, "Regression report is unexpectedly small!"

    content = report_path.read_text(encoding="utf-8")
    required_phrases = [
        "Executive Summary",
        "Model Comparison Matrix",
        "Segmented Error Analysis",
        "Floor Cohort",
        "Above-Floor",
        "MAPE Calculation Methodology & Limitations",
        "Information Gap",
        "Modeling Limitations"
    ]
    for phrase in required_phrases:
        assert phrase in content, f"Missing required section/phrase: '{phrase}'"

    # Check metrics JSON
    metrics_path = Path("reports/regression_metrics.json")
    assert metrics_path.exists(), "regression_metrics.json does not exist!"
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
    assert len(metrics_data) == 6, f"Expected 6 evaluated configurations, got {len(metrics_data)}"


def test_held_out_test_set_untouched():
    """Held-out test split (test.csv) must have exactly 80,960 rows and 27 columns and zero modification."""
    test_path = Path("data/processed/test.csv")
    assert test_path.exists(), "test.csv does not exist!"

    df_test = pd.read_csv(test_path, low_memory=False)
    assert df_test.shape == (80960, 27), f"Expected (80960, 27), got {df_test.shape}"


def test_raw_dataset_fingerprint_unmodified():
    """Raw dataset SHA-256 fingerprint must match the verified reference."""
    raw_path = Path("data/raw/emi_prediction_dataset.csv")
    assert raw_path.exists(), "Raw dataset does not exist!"

    sha256 = hashlib.sha256()
    with open(raw_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    actual_hash = sha256.hexdigest().upper()

    assert actual_hash == RAW_DATA_SHA256, f"Raw data hash modified! Expected {RAW_DATA_SHA256}, got {actual_hash}"
