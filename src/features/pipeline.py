"""
Feature preprocessing and pipeline definitions for EMIPredict AI.

Enforces critical ML safeguards and provides dual feature sets:
1. 'application' (24 features): Full application-time inputs genuinely submitted by
   an applicant (demographics, financial profile, requested loan amount, tenure, scenario).
   Strictly excludes internal debt limit (max_monthly_emi) and redundant family_size.
2. 'applicant_only' (21 features): Strictly applicant balance-sheet & capacity metrics,
   excluding all loan request parameters (requested_amount, requested_tenure, emi_scenario)
   to evaluate risk purely from customer intrinsic capacity without knowledge of loan terms.
3. max_monthly_emi is programmatically barred from classification features.
4. emi_eligibility is programmatically barred from regression features.
5. family_size is removed (perfectly collinear with dependents, r=1.0).
6. Missing values are imputed using median (numerical) and mode (categorical).
7. All transformers fit strictly on training folds to prevent data leakage.
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler

# Target variables
CLASSIFICATION_TARGET = "emi_eligibility"
REGRESSION_TARGET = "max_monthly_emi"

# Collinear feature removed to protect linear models
DROPPED_FEATURES = ["family_size"]

# Loan request features (disputed features evaluated for target-rule proxy risk)
LOAN_REQUEST_FEATURES = ["requested_amount", "requested_tenure", "emi_scenario"]

# --- Feature Set 1: Full Application-Time Schema (24 features) ---
CATEGORICAL_FEATURES_APPLICATION: List[str] = [
    "gender",
    "marital_status",
    "education",
    "employment_type",
    "company_type",
    "house_type",
    "existing_loans",
    "emi_scenario"
]

NUMERICAL_FEATURES_APPLICATION: List[str] = [
    "age",
    "monthly_salary",
    "years_of_employment",
    "monthly_rent",
    "dependents",
    "school_fees",
    "college_fees",
    "travel_expenses",
    "groceries_utilities",
    "other_monthly_expenses",
    "current_emi_amount",
    "credit_score",
    "bank_balance",
    "emergency_fund",
    "requested_amount",
    "requested_tenure"
]

FEATURE_NAMES_APPLICATION: List[str] = (
    CATEGORICAL_FEATURES_APPLICATION + NUMERICAL_FEATURES_APPLICATION
)

# --- Feature Set 2: Strict Applicant-Only Schema (21 features) ---
CATEGORICAL_FEATURES_APPLICANT_ONLY: List[str] = [
    c for c in CATEGORICAL_FEATURES_APPLICATION if c not in LOAN_REQUEST_FEATURES
]

NUMERICAL_FEATURES_APPLICANT_ONLY: List[str] = [
    n for n in NUMERICAL_FEATURES_APPLICATION if n not in LOAN_REQUEST_FEATURES
]

FEATURE_NAMES_APPLICANT_ONLY: List[str] = (
    CATEGORICAL_FEATURES_APPLICANT_ONLY + NUMERICAL_FEATURES_APPLICANT_ONLY
)

# Default aliases for backward compatibility
CATEGORICAL_FEATURES = CATEGORICAL_FEATURES_APPLICATION
NUMERICAL_FEATURES = NUMERICAL_FEATURES_APPLICATION
FEATURE_NAMES = FEATURE_NAMES_APPLICATION


class GenderStandardizer(BaseEstimator, TransformerMixin):
    """
    Standardizes gender text variants (e.g. 'M', 'male', 'MALE' -> 'Male'
    and 'F', 'female', 'FEMALE' -> 'Female') while preserving NaNs for imputation.
    """
    MAPPING = {
        "male": "Male",
        "m": "Male",
        "female": "Female",
        "f": "Female"
    }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        for col in X_df.columns:
            def clean_val(v):
                if pd.isna(v):
                    return v
                s = str(v).strip().lower()
                return self.MAPPING.get(s, str(v).title())
            X_df[col] = X_df[col].map(clean_val)
        return X_df.to_numpy() if isinstance(X, np.ndarray) else X_df


def get_classification_features(
    df: pd.DataFrame,
    feature_set: str = "application"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extracts X and y for classification.
    feature_set options:
      - 'application' (default, 24 features): includes loan request parameters (requested_amount, tenure, scenario).
      - 'applicant_only' (21 features): strictly excludes loan request parameters to avoid target-generation proxy rules.

    Strictly excludes max_monthly_emi and family_size to prevent data leakage and collinearity.
    """
    cols_to_drop = [CLASSIFICATION_TARGET, REGRESSION_TARGET] + DROPPED_FEATURES
    if feature_set == "applicant_only":
        cols_to_drop += LOAN_REQUEST_FEATURES
        expected_features = FEATURE_NAMES_APPLICANT_ONLY
    elif feature_set == "application":
        expected_features = FEATURE_NAMES_APPLICATION
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}. Must be 'application' or 'applicant_only'.")

    existing_drops = [c for c in cols_to_drop if c in df.columns]
    X = df.drop(columns=existing_drops).copy()
    y = df[CLASSIFICATION_TARGET].copy()

    # Integrity assertions
    assert REGRESSION_TARGET not in X.columns, f"Target leakage! {REGRESSION_TARGET} found in classification features."
    assert CLASSIFICATION_TARGET not in X.columns, f"Target {CLASSIFICATION_TARGET} found in feature matrix."
    for col in DROPPED_FEATURES:
        assert col not in X.columns, f"Collinear feature {col} not dropped."
    if feature_set == "applicant_only":
        for col in LOAN_REQUEST_FEATURES:
            assert col not in X.columns, f"Loan request feature {col} present in applicant_only set."

    # Verify expected schema
    missing_cols = set(expected_features) - set(X.columns)
    assert not missing_cols, f"Missing required features: {missing_cols}"

    return X[expected_features], y


def get_regression_features(
    df: pd.DataFrame,
    feature_set: str = "application"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extracts X and y for regression.
    Strictly excludes emi_eligibility and family_size to prevent target leakage and collinearity.
    """
    cols_to_drop = [CLASSIFICATION_TARGET, REGRESSION_TARGET] + DROPPED_FEATURES
    if feature_set == "applicant_only":
        cols_to_drop += LOAN_REQUEST_FEATURES
        expected_features = FEATURE_NAMES_APPLICANT_ONLY
    elif feature_set == "application":
        expected_features = FEATURE_NAMES_APPLICATION
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}. Must be 'application' or 'applicant_only'.")

    existing_drops = [c for c in cols_to_drop if c in df.columns]
    X = df.drop(columns=existing_drops).copy()
    y = df[REGRESSION_TARGET].copy()

    # Integrity assertions
    assert CLASSIFICATION_TARGET not in X.columns, f"Target leakage! {CLASSIFICATION_TARGET} found in regression features."
    assert REGRESSION_TARGET not in X.columns, f"Target {REGRESSION_TARGET} found in feature matrix."
    for col in DROPPED_FEATURES:
        assert col not in X.columns, f"Collinear feature {col} not dropped."
    if feature_set == "applicant_only":
        for col in LOAN_REQUEST_FEATURES:
            assert col not in X.columns, f"Loan request feature {col} present in applicant_only set."

    missing_cols = set(expected_features) - set(X.columns)
    assert not missing_cols, f"Missing required features: {missing_cols}"

    return X[expected_features], y


def build_preprocessor(
    feature_set: str = "application",
    scale: bool = False,
    scaler_type: str = "standard"
) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer matching the specified feature_set:
    - Categorical pipeline: GenderStandardizer -> SimpleImputer(most_frequent) -> OneHotEncoder(ignore)
    - Numerical pipeline: SimpleImputer(median) -> optional StandardScaler/RobustScaler
    """
    if feature_set == "applicant_only":
        cat_cols = CATEGORICAL_FEATURES_APPLICANT_ONLY
        num_cols = NUMERICAL_FEATURES_APPLICANT_ONLY
    elif feature_set == "application":
        cat_cols = CATEGORICAL_FEATURES_APPLICATION
        num_cols = NUMERICAL_FEATURES_APPLICATION
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}. Must be 'application' or 'applicant_only'.")

    cat_pipeline_steps = [
        ("gender_clean", GenderStandardizer()),
        ("cat_imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]
    cat_pipeline = Pipeline(cat_pipeline_steps)

    num_pipeline_steps = [
        ("num_imputer", SimpleImputer(strategy="median"))
    ]
    if scale:
        if scaler_type == "robust":
            num_pipeline_steps.append(("scaler", RobustScaler()))
        else:
            num_pipeline_steps.append(("scaler", StandardScaler()))

    num_pipeline = Pipeline(num_pipeline_steps)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)
        ],
        remainder="drop"
    )
    return preprocessor


def build_classification_pipeline(
    model: Optional[BaseEstimator] = None,
    feature_set: str = "application",
    scale: bool = False,
    scaler_type: str = "standard"
) -> Pipeline:
    """
    Builds an end-to-end classification pipeline for the specified feature set.
    Scaling is applied for linear models (LogisticRegression); tree models (RF, XGBoost) use scale=False.
    """
    preprocessor = build_preprocessor(feature_set=feature_set, scale=scale, scaler_type=scaler_type)
    steps = [("preprocessor", preprocessor)]
    if model is not None:
        steps.append(("model", model))
    return Pipeline(steps)


def build_regression_pipeline(
    model: Optional[BaseEstimator] = None,
    feature_set: str = "application",
    scale: bool = False,
    scaler_type: str = "standard"
) -> Pipeline:
    """
    Builds an end-to-end regression pipeline for the specified feature set.
    Scaling is applied for linear models (LinearRegression); tree models (RF, XGBoost) use scale=False.
    """
    preprocessor = build_preprocessor(feature_set=feature_set, scale=scale, scaler_type=scaler_type)
    steps = [("preprocessor", preprocessor)]
    if model is not None:
        steps.append(("model", model))
    return Pipeline(steps)
