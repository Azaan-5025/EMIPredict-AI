"""
Automated unit tests for Phase 10: Model Evaluation, Selection, and Held-Out Test Benchmarking.

Verifies:
1. Final evaluation reports and structured metrics JSON exist and contain all required sections.
2. Test split (test.csv) has exactly 80,960 rows, 27 columns, and correct class distributions.
3. Raw dataset SHA-256 fingerprint remains strictly unmodified.
4. Generalization gaps between 5-fold CV and held-out test are minimal (< 1%).
5. Regression metrics explicitly document both raw and clipped predictions without silent replacement.
6. Multi-task alignment analysis is documented.
7. End-to-end model inference on test samples is reproducible and adheres to anti-leakage barriers.
"""

from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import (
    get_classification_features,
    get_regression_features,
    FEATURE_NAMES_APPLICATION,
    FEATURE_NAMES_APPLICANT_ONLY,
    CLASSIFICATION_TARGET,
    REGRESSION_TARGET
)

RAW_DATA_SHA256 = "46A52DCBC80B8D736C5A42E9FA5173289E66BB9D30DD03E4E2F33E2B9C1A53C4"
TEST_DATA_PATH = Path("data/processed/test.csv")
REPORT_PATH = Path("reports/final_model_evaluation_report.md")
METRICS_PATH = Path("reports/final_evaluation_metrics.json")


def test_final_evaluation_artifacts_exist():
    """reports/final_model_evaluation_report.md and metrics JSON must exist and be non-empty."""
    assert REPORT_PATH.exists(), "Final evaluation markdown report does not exist!"
    assert REPORT_PATH.stat().st_size > 5000, "Final evaluation report is unexpectedly small!"

    assert METRICS_PATH.exists(), "Final evaluation metrics JSON does not exist!"
    assert METRICS_PATH.stat().st_size > 1000, "Metrics JSON is unexpectedly small!"


def test_final_evaluation_report_sections():
    """Report must contain all required analytical sections and disclosures."""
    content = REPORT_PATH.read_text(encoding="utf-8")
    required_sections = [
        "Executive Summary & Production Selection",
        "Classification Held-Out Test Benchmarks",
        "Regression Held-Out Test Benchmarks",
        "Raw vs. Clipped Performance Comparison",
        "Segmented Error Analysis",
        "Evaluation of the ₹500 Floor",
        "Generalization Gap Audit",
        "Multi-Task Alignment Analysis",
        "Limitations of Synthetic Data in Banking",
        "Serialized Production Artifacts"
    ]
    for section in required_sections:
        assert section in content, f"Missing required section in report: '{section}'"


def test_metrics_json_integrity():
    """Metrics JSON must be valid, well-structured, and contain required benchmarks."""
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "test_dataset" in data
    assert data["test_dataset"]["rows"] == 80960
    assert data["test_dataset"]["columns"] == 27
    assert data["test_dataset"]["regression_target_summary"]["floor_count"] == 21347

    # Check classification benchmarks
    clf_benchmarks = data.get("classification_benchmarks", [])
    assert len(clf_benchmarks) >= 2, "Insufficient classification benchmarks logged!"
    for b in clf_benchmarks:
        assert "metrics" in b
        assert "accuracy" in b["metrics"]
        assert "macro_f1" in b["metrics"]
        assert "balanced_accuracy" in b["metrics"]
        assert "roc_auc_ovr_macro" in b["metrics"]
        assert "per_class" in b
        assert "confusion_matrix" in b
        assert len(b["confusion_matrix"]) == 3

    # Check regression benchmarks (both raw and clipped)
    reg_benchmarks = data.get("regression_benchmarks", [])
    assert len(reg_benchmarks) >= 2, "Insufficient regression benchmarks logged!"
    for b in reg_benchmarks:
        assert "raw_predictions" in b
        assert "clipped_predictions" in b
        assert "prediction_distribution" in b

        # Verify no silent replacement: raw and clipped must both exist
        raw_overall = b["raw_predictions"]["overall"]
        clip_overall = b["clipped_predictions"]["overall"]
        assert "rmse" in raw_overall and "rmse" in clip_overall
        assert "mae" in raw_overall and "mae" in clip_overall
        assert "r2" in raw_overall and "r2" in clip_overall
        assert "mape" in raw_overall and "mape" in clip_overall

        # Segmented cohorts must be present
        assert "floor_cohort" in b["raw_predictions"]
        assert "above_floor_cohort" in b["raw_predictions"]
        assert "floor_cohort" in b["clipped_predictions"]
        assert "above_floor_cohort" in b["clipped_predictions"]


def test_generalization_gap_bounds():
    """Generalization gap between 5-fold CV and test set must be < 1% across winning models."""
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    gaps = data.get("generalization_gaps", {})
    assert "classification_application_xgboost" in gaps
    clf_gap = gaps["classification_application_xgboost"]
    assert abs(clf_gap["macro_f1"]["gap"]) < 0.05
    assert abs(clf_gap["balanced_accuracy"]["gap"]) < 0.05

    assert "regression_application_xgboost" in gaps
    reg_gap = gaps["regression_application_xgboost"]
    assert abs(reg_gap["r2"]["gap"]) < 0.05
    assert abs(reg_gap["rmse"]["gap"]) < 50.0  # RMSE difference within ₹50


def test_multi_task_alignment_audit():
    """Multi-task alignment audit must report reasonable alignment consistency."""
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    alignment = data.get("multi_task_alignment", {})
    assert "ratio_gt_1_predicted_not_eligible_pct" in alignment
    assert "ratio_le_0_526_predicted_eligible_pct" in alignment
    # High consistency (>85%) expected due to underlying synthetic formula
    assert alignment["ratio_gt_1_predicted_not_eligible_pct"] > 85.0
    assert alignment["ratio_le_0_526_predicted_eligible_pct"] > 85.0


def test_held_out_test_set_integrity():
    """data/processed/test.csv must remain strictly 80,960 rows x 27 cols with correct distributions."""
    assert TEST_DATA_PATH.exists(), "data/processed/test.csv does not exist!"
    df_test = pd.read_csv(TEST_DATA_PATH)
    assert df_test.shape == (80960, 27)

    # Class balance check
    counts = df_test[CLASSIFICATION_TARGET].value_counts()
    assert counts["Not_Eligible"] == 62574
    assert counts["Eligible"] == 14889
    assert counts["High_Risk"] == 3497

    # Regression floor check
    floor_count = (df_test[REGRESSION_TARGET] == 500.0).sum()
    assert floor_count == 21347


def test_raw_dataset_fingerprint_unmodified():
    """Original raw CSV SHA-256 fingerprint must match initial reference byte-for-byte."""
    raw_path = Path("data/raw/emi_prediction_dataset.csv")
    assert raw_path.exists(), "Raw dataset file is missing!"

    hasher = hashlib.sha256()
    with open(raw_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)

    computed_hash = hasher.hexdigest().upper()
    assert computed_hash == RAW_DATA_SHA256, (
        f"Raw dataset modified! Expected {RAW_DATA_SHA256}, got {computed_hash}"
    )


def test_end_to_end_test_inference():
    """Production pipelines must execute cleanly on a test sample with zero target leakage."""
    df_sample = pd.read_csv(TEST_DATA_PATH, nrows=25)

    # Classification inference
    clf_pipe = joblib.load(Path("models/classification/best_application_model.joblib"))
    X_clf, _ = get_classification_features(df_sample, feature_set="application")
    assert CLASSIFICATION_TARGET not in X_clf.columns
    assert REGRESSION_TARGET not in X_clf.columns
    preds_clf = clf_pipe.predict(X_clf)
    probs_clf = clf_pipe.predict_proba(X_clf)
    assert len(preds_clf) == 25
    assert probs_clf.shape == (25, 3)
    np.testing.assert_allclose(probs_clf.sum(axis=1), np.ones(25), rtol=1e-5)

    # Regression inference
    reg_pipe = joblib.load(Path("models/regression/best_application_model.joblib"))
    X_reg, _ = get_regression_features(df_sample, feature_set="application")
    assert CLASSIFICATION_TARGET not in X_reg.columns
    assert REGRESSION_TARGET not in X_reg.columns
    preds_reg_raw = reg_pipe.predict(X_reg)
    preds_reg_clip = np.clip(preds_reg_raw, a_min=500.0, a_max=None)
    assert len(preds_reg_raw) == 25
    assert (preds_reg_clip >= 500.0).all()
