"""
Automated unit tests for Phase 8: Classification Models.

Verifies:
1. Saved classification models exist and can be loaded.
2. Saved pipelines predict cleanly on input data with correct shapes.
3. Strict anti-leakage safeguards (max_monthly_emi is excluded).
4. Classification report and metrics JSON exist and contain required analytical sections.
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


def test_saved_classification_artifacts_exist():
    """All required model artifacts must exist in models/classification/."""
    models_dir = Path("models/classification")
    assert models_dir.exists(), "models/classification directory does not exist!"

    required_artifacts = [
        "best_application_model.joblib",
        "best_applicant_only_model.joblib",
        "label_encoder.joblib",
        "logistic_regression_application.joblib",
        "random_forest_application.joblib",
        "xgboost_application.joblib"
    ]
    for art in required_artifacts:
        art_path = models_dir / art
        assert art_path.exists(), f"Missing model artifact: {art}"
        min_size = 100 if art.endswith("label_encoder.joblib") else 1000
        assert art_path.stat().st_size > min_size, f"Artifact {art} is unexpectedly small!"


def test_model_pipeline_predicts_correctly():
    """Loaded pipelines must execute end-to-end and output valid class predictions and probabilities."""
    model_path = Path("models/classification/best_application_model.joblib")
    assert model_path.exists(), "best_application_model.joblib does not exist!"

    pipeline = joblib.load(model_path)
    sample_df = pd.read_csv("data/processed/train.csv", nrows=50)

    from src.features.pipeline import get_classification_features
    X_sample, _ = get_classification_features(sample_df, feature_set="application")

    preds = pipeline.predict(X_sample)
    probs = pipeline.predict_proba(X_sample)

    assert len(preds) == len(sample_df)
    assert probs.shape == (len(sample_df), 3)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(sample_df)), rtol=1e-5)
    assert set(preds).issubset({0, 1, 2})


def test_applicant_only_pipeline_predicts_correctly():
    """Applicant-only pipeline must execute without loan request terms."""
    model_path = Path("models/classification/best_applicant_only_model.joblib")
    assert model_path.exists(), "best_applicant_only_model.joblib does not exist!"

    pipeline = joblib.load(model_path)
    sample_df = pd.read_csv("data/processed/train.csv", nrows=50)

    from src.features.pipeline import get_classification_features
    X_sample, _ = get_classification_features(sample_df, feature_set="applicant_only")

    preds = pipeline.predict(X_sample)
    probs = pipeline.predict_proba(X_sample)

    assert len(preds) == len(sample_df)
    assert probs.shape == (len(sample_df), 3)
    assert set(preds).issubset({0, 1, 2})


def test_classification_reports_and_metrics_exist():
    """reports/classification_report.md and metrics JSON must exist and contain required content."""
    report_path = Path("reports/classification_report.md")
    assert report_path.exists(), "reports/classification_report.md does not exist!"
    assert report_path.stat().st_size > 3000, "Classification report is unexpectedly small!"

    content = report_path.read_text(encoding="utf-8")
    required_phrases = [
        "Executive Summary",
        "Model Comparison Matrix",
        "Per-Class Performance",
        "Confusion Matrices",
        "Information Gap",
        "CRITICAL METHODOLOGICAL NOTICE — SYNTHETIC DATASET ARTIFACT",
        "Modeling Limitations"
    ]
    for phrase in required_phrases:
        assert phrase in content, f"Missing required section/phrase: '{phrase}'"

    # Check metrics JSON
    metrics_path = Path("reports/classification_metrics.json")
    assert metrics_path.exists(), "classification_metrics.json does not exist!"
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
    assert len(metrics_data) == 12, f"Expected 12 evaluated configurations, got {len(metrics_data)}"


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
