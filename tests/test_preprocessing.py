"""
Automated unit tests for Phase 6: Preprocessing & Data Preparation.

Verifies:
1. Strict target exclusion for classification (no max_monthly_emi).
2. Strict target exclusion for regression (no emi_eligibility).
3. Redundant collinear feature exclusion (family_size dropped).
4. Feature schema consistency (24 input features: 8 cat + 16 num).
5. Reproducibility of train/test splits (random_state=42).
6. Stratification preservation across train and test sets.
7. Handling of missing values (no remaining NaNs after pipeline).
8. Gender standardization for mixed-case and abbreviated entries.
9. Handling of unseen/out-of-vocabulary categorical levels in test data.
10. Strict absence of train-test data leakage during preprocessing.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.features.pipeline import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    DROPPED_FEATURES,
    FEATURE_NAMES,
    CLASSIFICATION_TARGET,
    REGRESSION_TARGET,
    GenderStandardizer,
    get_classification_features,
    get_regression_features,
    build_preprocessor,
    build_classification_pipeline,
    build_regression_pipeline
)
from src.data.split_data import split_dataset, load_train_test_data

@pytest.fixture(scope="module")
def sample_data():
    """Loads a small sample from the cleaned dataset for fast testing."""
    df = pd.read_csv("data/processed/emi_cleaned.csv", nrows=1000, low_memory=False)
    return df

@pytest.fixture(scope="module")
def train_test_data():
    """Loads the official train/test splits."""
    return load_train_test_data()

def test_target_exclusion_classification(sample_data):
    """max_monthly_emi and family_size must never be in classification features."""
    X, y = get_classification_features(sample_data)

    assert REGRESSION_TARGET not in X.columns, f"{REGRESSION_TARGET} leaked into classification features!"
    assert CLASSIFICATION_TARGET not in X.columns, f"{CLASSIFICATION_TARGET} retained in feature matrix!"
    assert "family_size" not in X.columns, "family_size must be dropped!"
    assert "dependents" in X.columns, "dependents must be retained!"
    assert y.name == CLASSIFICATION_TARGET

def test_target_exclusion_regression(sample_data):
    """emi_eligibility and family_size must never be in regression features."""
    X, y = get_regression_features(sample_data)

    assert CLASSIFICATION_TARGET not in X.columns, f"{CLASSIFICATION_TARGET} leaked into regression features!"
    assert REGRESSION_TARGET not in X.columns, f"{REGRESSION_TARGET} retained in feature matrix!"
    assert "family_size" not in X.columns, "family_size must be dropped!"
    assert "dependents" in X.columns, "dependents must be retained!"
    assert y.name == REGRESSION_TARGET

def test_feature_schema_consistency(sample_data):
    """Feature matrix must contain exactly 24 features (8 categorical + 16 numerical)."""
    X_cls, _ = get_classification_features(sample_data)
    X_reg, _ = get_regression_features(sample_data)

    assert len(CATEGORICAL_FEATURES) == 8
    assert len(NUMERICAL_FEATURES) == 16
    assert len(FEATURE_NAMES) == 24
    assert list(X_cls.columns) == FEATURE_NAMES
    assert list(X_reg.columns) == FEATURE_NAMES

def test_reproducible_splitting(tmp_path):
    """Splitting with seed 42 must generate identical datasets every time."""
    train1, test1 = split_dataset(output_dir=tmp_path / "run1", random_state=42)
    train2, test2 = split_dataset(output_dir=tmp_path / "run2", random_state=42)

    pd.testing.assert_frame_equal(train1, train2)
    pd.testing.assert_frame_equal(test1, test2)

def test_stratification_balance(train_test_data):
    """Train and test splits must preserve class proportions within 0.1% margin."""
    train_df, test_df = train_test_data

    train_pct = train_df[CLASSIFICATION_TARGET].value_counts(normalize=True) * 100
    test_pct = test_df[CLASSIFICATION_TARGET].value_counts(normalize=True) * 100

    for cls_label in ["Not_Eligible", "Eligible", "High_Risk"]:
        diff = abs(train_pct[cls_label] - test_pct[cls_label])
        assert diff < 0.1, f"Stratification mismatch for {cls_label}: train={train_pct[cls_label]:.2f}%, test={test_pct[cls_label]:.2f}%"

def test_gender_standardization():
    """GenderStandardizer must cleanly map variations to Male and Female."""
    standardizer = GenderStandardizer()
    inputs = pd.DataFrame({"gender": ["Male", "male", "MALE", "M", "Female", "female", "FEMALE", "F", None]})
    transformed = standardizer.transform(inputs)

    expected = ["Male", "Male", "Male", "Male", "Female", "Female", "Female", "Female", None]
    for actual, exp in zip(transformed["gender"], expected):
        if exp is None:
            assert pd.isna(actual)
        else:
            assert actual == exp

def test_missing_value_handling():
    """Preprocessor must cleanly impute all missing values with no residual NaNs."""
    df_nan = pd.DataFrame({
        "gender": ["Male", np.nan, "female"],
        "marital_status": ["Married", "Single", np.nan],
        "education": [np.nan, "Graduate", "Professional"],
        "employment_type": ["Private", "Government", np.nan],
        "company_type": [np.nan, "MNC", "Startup"],
        "house_type": ["Rented", np.nan, "Own"],
        "existing_loans": ["Yes", "No", np.nan],
        "emi_scenario": ["Vehicle EMI", np.nan, "Personal Loan EMI"],
        "age": [30.0, np.nan, 45.0],
        "monthly_salary": [np.nan, 50000.0, 75000.0],
        "years_of_employment": [2.0, np.nan, 8.0],
        "monthly_rent": [np.nan, 12000.0, 0.0],
        "dependents": [1.0, 2.0, np.nan],
        "school_fees": [0.0, np.nan, 1500.0],
        "college_fees": [np.nan, 0.0, 2000.0],
        "travel_expenses": [2000.0, np.nan, 4000.0],
        "groceries_utilities": [np.nan, 10000.0, 15000.0],
        "other_monthly_expenses": [3000.0, np.nan, 6000.0],
        "current_emi_amount": [np.nan, 0.0, 5000.0],
        "credit_score": [np.nan, 720.0, 650.0],
        "bank_balance": [100000.0, np.nan, 250000.0],
        "emergency_fund": [np.nan, 30000.0, 60000.0],
        "requested_amount": [200000.0, 500000.0, np.nan],
        "requested_tenure": [12.0, np.nan, 36.0]
    })

    preprocessor = build_preprocessor(scale=True)
    out = preprocessor.fit_transform(df_nan)

    assert not np.isnan(out).any(), "Residual NaNs found in preprocessed output!"
    assert not np.isinf(out).any(), "Infinities found in preprocessed output!"
    assert out.shape[0] == 3

def test_unseen_categories_handling(sample_data):
    """OneHotEncoder must ignore unseen levels in test sets without throwing errors."""
    X_train, _ = get_classification_features(sample_data.iloc[:500])
    X_test, _ = get_classification_features(sample_data.iloc[500:600])

    # Inject unseen categories into test set
    X_test_unseen = X_test.copy()
    X_test_unseen.loc[X_test_unseen.index[0], "education"] = "Doctorate_Degree"
    X_test_unseen.loc[X_test_unseen.index[1], "company_type"] = "Autonomous_Org"

    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    # Should transform without exception
    transformed_test = preprocessor.transform(X_test_unseen)
    assert transformed_test.shape[0] == len(X_test)
    assert not np.isnan(transformed_test).any()

def test_no_train_test_data_leakage(sample_data):
    """
    Imputer medians and scaler means must strictly be computed from train data.
    Evaluating test data must not modify or influence the fitted parameters.
    """
    X, _ = get_classification_features(sample_data)
    X_train = X.iloc[:700].copy()
    X_test = X.iloc[700:].copy()

    preprocessor = build_preprocessor(scale=True)
    preprocessor.fit(X_train)

    num_imputer = preprocessor.named_transformers_["num"].named_steps["num_imputer"]
    scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]

    train_medians = num_imputer.statistics_.copy()
    train_means = scaler.mean_.copy()

    # Verify imputer medians match X_train medians exactly
    expected_medians = X_train[NUMERICAL_FEATURES].median().to_numpy()
    np.testing.assert_allclose(train_medians, expected_medians, rtol=1e-5)

    # Transform test set
    preprocessor.transform(X_test)

    # Parameters must remain completely unchanged
    np.testing.assert_array_equal(num_imputer.statistics_, train_medians)
    np.testing.assert_array_equal(scaler.mean_, train_means)


def test_dual_classification_feature_sets(sample_data):
    """Verifies that both 'application' and 'applicant_only' feature sets work correctly."""
    from src.features.pipeline import (
        FEATURE_NAMES_APPLICATION,
        FEATURE_NAMES_APPLICANT_ONLY,
        LOAN_REQUEST_FEATURES
    )
    # Application schema (24 features)
    X_app, y_app = get_classification_features(sample_data, feature_set="application")
    assert len(X_app.columns) == 24
    assert list(X_app.columns) == FEATURE_NAMES_APPLICATION
    assert "max_monthly_emi" not in X_app.columns
    assert "family_size" not in X_app.columns
    for f in LOAN_REQUEST_FEATURES:
        assert f in X_app.columns

    # Applicant-only schema (21 features)
    X_pure, y_pure = get_classification_features(sample_data, feature_set="applicant_only")
    assert len(X_pure.columns) == 21
    assert list(X_pure.columns) == FEATURE_NAMES_APPLICANT_ONLY
    assert "max_monthly_emi" not in X_pure.columns
    assert "family_size" not in X_pure.columns
    for f in LOAN_REQUEST_FEATURES:
        assert f not in X_pure.columns, f"{f} must be excluded in applicant_only set!"


def test_dual_regression_feature_sets(sample_data):
    """Verifies that both regression schemas work and exclude emi_eligibility and family_size."""
    X_app, y_app = get_regression_features(sample_data, feature_set="application")
    assert len(X_app.columns) == 24
    assert "emi_eligibility" not in X_app.columns
    assert "family_size" not in X_app.columns

    X_pure, y_pure = get_regression_features(sample_data, feature_set="applicant_only")
    assert len(X_pure.columns) == 21
    assert "emi_eligibility" not in X_pure.columns
    assert "family_size" not in X_pure.columns
    for f in ["requested_amount", "requested_tenure", "emi_scenario"]:
        assert f not in X_pure.columns


def test_preprocessor_dual_feature_sets(sample_data):
    """Verifies that preprocessors build and transform for both feature sets."""
    X_app, _ = get_classification_features(sample_data, feature_set="application")
    prep_app = build_preprocessor(feature_set="application", scale=False)
    out_app = prep_app.fit_transform(X_app)
    assert out_app.shape[0] == len(sample_data)
    assert not np.isnan(out_app).any()

    X_pure, _ = get_classification_features(sample_data, feature_set="applicant_only")
    prep_pure = build_preprocessor(feature_set="applicant_only", scale=False)
    out_pure = prep_pure.fit_transform(X_pure)
    assert out_pure.shape[0] == len(sample_data)
    assert not np.isnan(out_pure).any()


def test_invalid_feature_set_raises(sample_data):
    """Invalid feature_set name must raise ValueError."""
    with pytest.raises(ValueError, match="Unknown feature_set"):
        get_classification_features(sample_data, feature_set="invalid_set")

    with pytest.raises(ValueError, match="Unknown feature_set"):
        build_preprocessor(feature_set="invalid_set")
