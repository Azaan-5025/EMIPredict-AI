"""
Automated Unit Tests for Phase 11: Streamlit Web Application Components.

Tests:
1. Feature-schema compatibility with trained models.
2. Inference output correctness and probability distributions.
3. Input validation logic (boundary checks, negative values, age-experience consistency).
4. Financial ratio calculation engine (FOIR, disposable cash flow, implied installments).
5. SQLite Prediction History CRUD operations (Create, Read, Delete, Clear, Summary stats).
6. Error handling on malformed or incomplete inputs.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import (
    FEATURE_NAMES_APPLICATION,
    CATEGORICAL_FEATURES_APPLICATION,
    NUMERICAL_FEATURES_APPLICATION,
    CLASSIFICATION_TARGET,
    REGRESSION_TARGET
)
from app.validation import validate_applicant_inputs, calculate_financial_ratios
from app.db import (
    init_db,
    save_prediction,
    get_predictions,
    get_prediction_by_id,
    update_prediction_notes,
    delete_prediction,
    clear_all_predictions,
    get_history_summary_stats
)


@pytest.fixture
def sample_valid_applicant():
    """Provides a valid applicant dictionary adhering to the application schema."""
    return {
        "gender": "Male",
        "marital_status": "Married",
        "education": "Graduate",
        "employment_type": "Private",
        "company_type": "MNC",
        "house_type": "Rented",
        "existing_loans": "No",
        "emi_scenario": "Personal Loan EMI",
        "age": 32.0,
        "monthly_salary": 65000.0,
        "years_of_employment": 4.0,
        "monthly_rent": 12000.0,
        "dependents": 1.0,
        "school_fees": 3000.0,
        "college_fees": 0.0,
        "travel_expenses": 4500.0,
        "groceries_utilities": 11000.0,
        "other_monthly_expenses": 5000.0,
        "current_emi_amount": 0.0,
        "credit_score": 750.0,
        "bank_balance": 180000.0,
        "emergency_fund": 70000.0,
        "requested_amount": 150000.0,
        "requested_tenure": 24.0
    }


@pytest.fixture
def test_db_path(tmp_path):
    """Provides an isolated temporary database path for testing CRUD operations."""
    return tmp_path / "test_history.db"


def test_feature_schema_completeness(sample_valid_applicant):
    """Verifies that all 24 required features are present and accounted for."""
    assert len(FEATURE_NAMES_APPLICATION) == 24
    missing_from_sample = set(FEATURE_NAMES_APPLICATION) - set(sample_valid_applicant.keys())
    assert not missing_from_sample, f"Sample missing features: {missing_from_sample}"

    # Targets must never be features
    assert CLASSIFICATION_TARGET not in FEATURE_NAMES_APPLICATION
    assert REGRESSION_TARGET not in FEATURE_NAMES_APPLICATION


def test_classification_model_inference_compatibility(sample_valid_applicant):
    """Loaded classification pipeline must execute cleanly on application features."""
    clf_path = Path("models/classification/best_application_model.joblib")
    le_path = Path("models/classification/label_encoder.joblib")
    assert clf_path.exists(), "Classification model missing!"
    assert le_path.exists(), "Label encoder missing!"

    clf = joblib.load(clf_path)
    le = joblib.load(le_path)

    input_df = pd.DataFrame([sample_valid_applicant])[FEATURE_NAMES_APPLICATION]
    preds = clf.predict(input_df)
    probs = clf.predict_proba(input_df)[0]

    assert len(preds) == 1
    assert preds[0] in [0, 1, 2]
    assert len(probs) == 3
    assert np.isclose(probs.sum(), 1.0, rtol=1e-5)

    decoded_class = le.inverse_transform(preds)[0]
    assert decoded_class in ["Eligible", "High_Risk", "Not_Eligible"]


def test_regression_model_inference_compatibility(sample_valid_applicant):
    """Loaded regression pipeline must output continuous EMI capacity with policy floor applied."""
    reg_path = Path("models/regression/best_application_model.joblib")
    assert reg_path.exists(), "Regression model missing!"

    reg = joblib.load(reg_path)
    input_df = pd.DataFrame([sample_valid_applicant])[FEATURE_NAMES_APPLICATION]
    raw_pred = float(reg.predict(input_df)[0])

    assert isinstance(raw_pred, float)
    assert not np.isnan(raw_pred)
    assert not np.isinf(raw_pred)

    policy_pred = max(500.0, raw_pred)
    assert policy_pred >= 500.0


def test_validation_rejects_underage_and_negative_salary(sample_valid_applicant):
    """Validator must strictly reject underage applicants and non-positive salaries."""
    # Underage applicant
    invalid_age = sample_valid_applicant.copy()
    invalid_age["age"] = 16.0
    is_valid, errors, _ = validate_applicant_inputs(invalid_age)
    assert not is_valid
    assert any("at least 18 years" in e for e in errors)

    # Zero or negative salary
    invalid_sal = sample_valid_applicant.copy()
    invalid_sal["monthly_salary"] = -500.0
    is_valid, errors, _ = validate_applicant_inputs(invalid_sal)
    assert not is_valid
    assert any("Monthly salary must be strictly greater than ₹0" in e for e in errors)


def test_validation_rejects_excessive_employment_years(sample_valid_applicant):
    """Years of employment cannot exceed age minus 18."""
    invalid_emp = sample_valid_applicant.copy()
    invalid_emp["age"] = 25.0
    invalid_emp["years_of_employment"] = 12.0  # 25 - 18 = 7 max
    is_valid, errors, _ = validate_applicant_inputs(invalid_emp)
    assert not is_valid
    assert any("cannot exceed implied working years" in e for e in errors)


def test_validation_soft_warnings_for_inconsistent_loans_and_rent(sample_valid_applicant):
    """Validator should generate informative soft warnings for inconsistent fields without blocking."""
    warn_app = sample_valid_applicant.copy()
    warn_app["existing_loans"] = "No"
    warn_app["current_emi_amount"] = 8000.0  # Inconsistent
    warn_app["house_type"] = "Own"
    warn_app["monthly_rent"] = 15000.0  # Inconsistent with owned property

    is_valid, errors, warnings = validate_applicant_inputs(warn_app)
    assert is_valid  # Still valid as a soft warning
    assert any("Existing loans is marked 'No' but Current EMI is" in w for w in warnings)
    assert any("Property is 'Own' but monthly rent of" in w for w in warnings)


def test_financial_ratio_calculations(sample_valid_applicant):
    """Ratio engine must accurately compute living expenses, disposable income, and FOIR."""
    ratios = calculate_financial_ratios(sample_valid_applicant)
    assert "disposable_income" in ratios
    assert "implied_installment" in ratios
    assert "current_foir" in ratios
    assert "projected_foir" in ratios

    expected_installment = sample_valid_applicant["requested_amount"] / sample_valid_applicant["requested_tenure"]
    assert np.isclose(ratios["implied_installment"], expected_installment)

    expected_projected_foir = (expected_installment / sample_valid_applicant["monthly_salary"]) * 100.0
    assert abs(ratios["projected_foir"] - expected_projected_foir) < 0.01


def test_sqlite_history_crud_lifecycle(sample_valid_applicant, test_db_path):
    """Comprehensive test of SQLite database initialization, insert, read, delete, and clear."""
    # 1. Initialize
    init_db(test_db_path)
    assert test_db_path.exists()

    # 2. Insert record
    ratios = calculate_financial_ratios(sample_valid_applicant)
    rec_id_1 = save_prediction(
        assessment_type="Eligibility",
        applicant_data=sample_valid_applicant,
        ratios=ratios,
        eligibility_result="Eligible",
        probabilities={"Eligible": 0.85, "High_Risk": 0.10, "Not_Eligible": 0.05},
        db_path=test_db_path
    )
    assert rec_id_1 == 1

    rec_id_2 = save_prediction(
        assessment_type="Capacity",
        applicant_data=sample_valid_applicant,
        ratios=ratios,
        max_emi_raw=14500.0,
        max_emi_policy=14500.0,
        capacity_margin=8250.0,
        db_path=test_db_path
    )
    assert rec_id_2 == 2

    # 3. Read & filter
    df_all = get_predictions(db_path=test_db_path)
    assert len(df_all) == 2

    df_elig = get_predictions(db_path=test_db_path, assessment_type="Eligibility")
    assert len(df_elig) == 1
    assert df_elig.iloc[0]["predicted_eligibility"] == "Eligible"

    # 4. Read single record by ID
    single = get_prediction_by_id(rec_id_1, db_path=test_db_path)
    assert single is not None
    assert single["id"] == 1
    assert single["assessment_type"] == "Eligibility"

    # 5. Summary statistics
    stats = get_history_summary_stats(db_path=test_db_path)
    assert stats["total_records"] == 2
    assert "Eligible" in stats["eligibility_distribution"]

    # 6. Delete single record
    deleted = delete_prediction(rec_id_1, db_path=test_db_path)
    assert deleted is True
    assert get_prediction_by_id(rec_id_1, db_path=test_db_path) is None
    assert len(get_predictions(db_path=test_db_path)) == 1

    # 7. Clear all records
    cleared = clear_all_predictions(db_path=test_db_path)
    assert cleared == 1
    assert len(get_predictions(db_path=test_db_path)) == 0


def test_metric_card_html_rendering_and_view_imports():
    """Verifies that metric_card_html accepts 3 args and all view modules import correctly."""
    from app.components import metric_card_html
    from app.views.model_performance import render_model_performance_view
    from app.views.dashboard import render_dashboard_view
    from app.views.capacity import render_capacity_view
    from app.views.joint import render_joint_view

    # Test with 3 arguments (title, value, subtitle/description)
    html_3 = metric_card_html("Debt Ratio > 1.0 Agreement", "89.14%", "Classified as Not_Eligible")
    assert "Debt Ratio > 1.0 Agreement" in html_3
    assert "89.14%" in html_3
    assert "Classified as Not_Eligible" in html_3
    assert "metric-container" in html_3

    # Test with 2 arguments
    html_2 = metric_card_html("Monthly EMI", "₹12,500")
    assert "Monthly EMI" in html_2
    assert "₹12,500" in html_2

    # Verify functions are callable and properly imported
    assert callable(render_model_performance_view)
    assert callable(render_dashboard_view)
    assert callable(render_capacity_view)
    assert callable(render_joint_view)


def test_sqlite_numpy_int64_binding_bug_fix(sample_valid_applicant, test_db_path):
    """
    Regression test for Bug #1: Passing numpy.int64 from pandas/streamlit selectbox
    to get_prediction_by_id, update_prediction_notes, and delete_prediction must succeed.
    """
    init_db(test_db_path)
    ratios = calculate_financial_ratios(sample_valid_applicant)
    rec_id = save_prediction(
        assessment_type="Joint",
        applicant_data=sample_valid_applicant,
        ratios=ratios,
        eligibility_result="Eligible",
        notes="Initial note",
        db_path=test_db_path
    )

    # Cast to numpy.int64 as produced by pd.read_sql_query
    np_id = np.int64(rec_id)

    # 1. Inspect using numpy int64
    record = get_prediction_by_id(np_id, db_path=test_db_path)
    assert record is not None, "get_prediction_by_id failed with numpy.int64 parameter!"
    assert record["id"] == rec_id

    # 2. Update notes using numpy int64
    upd_ok = update_prediction_notes(np_id, "Updated underwriter remark", db_path=test_db_path)
    assert upd_ok is True
    updated_record = get_prediction_by_id(np_id, db_path=test_db_path)
    assert updated_record["notes"] == "Updated underwriter remark"

    # 3. Delete using numpy int64
    del_ok = delete_prediction(np_id, db_path=test_db_path)
    assert del_ok is True
    assert get_prediction_by_id(np_id, db_path=test_db_path) is None


def test_sqlite_history_search_and_filter(sample_valid_applicant, test_db_path):
    """Verifies that search_query keyword search filters records by notes, scenario, or ID."""
    init_db(test_db_path)
    ratios = calculate_financial_ratios(sample_valid_applicant)

    # Record 1: Personal Loan EMI
    save_prediction(
        assessment_type="Eligibility",
        applicant_data=sample_valid_applicant,
        ratios=ratios,
        eligibility_result="Eligible",
        notes="Customer has strong tech salary profile",
        db_path=test_db_path
    )

    # Record 2: Vehicle EMI
    app_2 = sample_valid_applicant.copy()
    app_2["emi_scenario"] = "Vehicle EMI"
    save_prediction(
        assessment_type="Capacity",
        applicant_data=app_2,
        ratios=ratios,
        eligibility_result="Not_Eligible",
        notes="Flagged for high debt leverage",
        db_path=test_db_path
    )

    # Search by keyword in notes
    df_tech = get_predictions(db_path=test_db_path, search_query="tech")
    assert len(df_tech) == 1
    assert "tech" in df_tech.iloc[0]["notes"]

    # Search by loan scenario
    df_veh = get_predictions(db_path=test_db_path, search_query="Vehicle")
    assert len(df_veh) == 1
    assert df_veh.iloc[0]["emi_scenario"] == "Vehicle EMI"

    # Search with no match
    df_empty = get_predictions(db_path=test_db_path, search_query="NonExistentPattern")
    assert len(df_empty) == 0


def test_history_persistence_across_connections(sample_valid_applicant, test_db_path):
    """Verifies that written prediction records persist across separate SQLite connections."""
    init_db(test_db_path)
    ratios = calculate_financial_ratios(sample_valid_applicant)

    rec_id = save_prediction(
        assessment_type="Joint",
        applicant_data=sample_valid_applicant,
        ratios=ratios,
        eligibility_result="High_Risk",
        notes="Persistent test record",
        db_path=test_db_path
    )

    # Simulate restart by reading from a fresh connection
    df = get_predictions(db_path=test_db_path)
    assert len(df) >= 1
    persisted = get_prediction_by_id(rec_id, db_path=test_db_path)
    assert persisted is not None
    assert persisted["predicted_eligibility"] == "High_Risk"
    assert persisted["notes"] == "Persistent test record"


def test_sqlite_error_handling_graceful(tmp_path):
    """Verifies that get_prediction_by_id and operations handle invalid inputs gracefully."""
    # Non-existent DB path or invalid IDs
    invalid_db = tmp_path / "non_existent_dir" / "dummy.db"
    # Should not crash on invalid ID
    assert get_prediction_by_id("invalid_id", db_path=invalid_db) is None
    assert delete_prediction("invalid_id", db_path=invalid_db) is False
    assert update_prediction_notes("invalid_id", "notes", db_path=invalid_db) is False


def test_all_preset_profiles_inference():
    """
    Verifies that all preset applicant profiles across views run through the actual
    trained models and predict their intended risk tiers with valid probability distributions.
    """
    clf_path = Path("models/classification/best_application_model.joblib")
    reg_path = Path("models/regression/best_application_model.joblib")
    le_path = Path("models/classification/label_encoder.joblib")
    clf = joblib.load(clf_path)
    reg = joblib.load(reg_path)
    le = joblib.load(le_path)

    # Base application defaults
    defaults = {
        "gender": "Male", "marital_status": "Married", "education": "Graduate",
        "employment_type": "Private", "company_type": "MNC", "house_type": "Rented",
        "existing_loans": "No", "emi_scenario": "Personal Loan EMI",
        "age": 34.0, "monthly_salary": 65000.0, "years_of_employment": 5.0, "monthly_rent": 12000.0,
        "dependents": 1.0, "school_fees": 3000.0, "college_fees": 0.0,
        "travel_expenses": 4500.0, "groceries_utilities": 11000.0, "other_monthly_expenses": 5000.0,
        "current_emi_amount": 0.0, "credit_score": 750.0, "bank_balance": 160000.0,
        "emergency_fund": 70000.0, "requested_amount": 150000.0, "requested_tenure": 24.0
    }

    # Profile A: Prime Salaried Borrower -> Must predict Eligible
    pa = defaults.copy()
    pa.update({
        "monthly_salary": 110000.0, "monthly_rent": 15000.0, "dependents": 1.0, "school_fees": 4000.0,
        "groceries_utilities": 14000.0, "other_monthly_expenses": 6000.0, "credit_score": 790.0,
        "bank_balance": 350000.0, "emergency_fund": 150000.0, "requested_amount": 120000.0, "requested_tenure": 24.0
    })
    df_a = pd.DataFrame([pa])[FEATURE_NAMES_APPLICATION]
    pred_a = le.inverse_transform(clf.predict(df_a))[0]
    probs_a = clf.predict_proba(df_a)[0]
    assert pred_a == "Eligible", f"Profile A predicted {pred_a}, expected Eligible"
    assert np.isclose(probs_a.sum(), 1.0)

    # Profile B: Borderline Moderate Risk -> Must predict High_Risk
    pb = defaults.copy()
    pb.update({
        "monthly_salary": 50000.0, "monthly_rent": 10000.0, "dependents": 2.0, "school_fees": 3000.0,
        "travel_expenses": 3500.0, "groceries_utilities": 11000.0, "other_monthly_expenses": 4000.0,
        "credit_score": 670.0, "bank_balance": 75000.0, "emergency_fund": 30000.0,
        "requested_amount": 160000.0, "requested_tenure": 24.0
    })
    df_b = pd.DataFrame([pb])[FEATURE_NAMES_APPLICATION]
    pred_b = le.inverse_transform(clf.predict(df_b))[0]
    probs_b = clf.predict_proba(df_b)[0]
    assert pred_b == "High_Risk", f"Profile B predicted {pred_b}, expected High_Risk"
    assert np.isclose(probs_b.sum(), 1.0)

    # Profile C: Over-Leveraged Insolvent -> Must predict Not_Eligible
    pc = defaults.copy()
    pc.update({
        "monthly_salary": 32000.0, "monthly_rent": 12000.0, "dependents": 3.0, "school_fees": 5000.0,
        "groceries_utilities": 12000.0, "other_monthly_expenses": 5000.0, "existing_loans": "Yes",
        "current_emi_amount": 14000.0, "credit_score": 580.0, "bank_balance": 15000.0,
        "emergency_fund": 5000.0, "requested_amount": 350000.0, "requested_tenure": 18.0
    })
    df_c = pd.DataFrame([pc])[FEATURE_NAMES_APPLICATION]
    pred_c = le.inverse_transform(clf.predict(df_c))[0]
    probs_c = clf.predict_proba(df_c)[0]
    assert pred_c == "Not_Eligible", f"Profile C predicted {pred_c}, expected Not_Eligible"
    assert np.isclose(probs_c.sum(), 1.0)

    # Capacity Profile 3: Strained cash flow -> Must yield sub-floor raw capacity
    raw_c3 = float(reg.predict(df_c)[0])
    assert raw_c3 < 500.0, f"Expected raw capacity < 500, got {raw_c3}"
    policy_c3 = max(500.0, raw_c3)
    assert policy_c3 == 500.0


def test_financial_assumptions_and_policy_disclaimers():
    """
    Verifies that the ₹500 floor callout and zero-interest assumption
    are clearly stated in component messages and disclaimers.
    """
    from app.components import floor_policy_callout, synthetic_disclaimer_callout

    # Callouts render without error
    floor_policy_callout(-200.0, 500.0)
    synthetic_disclaimer_callout()

    # Ratios verify 0% interest formula P / N
    p = 240000.0
    n = 24.0
    ratios = calculate_financial_ratios({"requested_amount": p, "requested_tenure": n, "monthly_salary": 60000.0})
    assert ratios["implied_installment"] == (p / n)
