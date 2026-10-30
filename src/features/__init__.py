"""
Feature engineering, schema definition, and scikit-learn preprocessing pipelines.
"""
from src.features.pipeline import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    DROPPED_FEATURES,
    FEATURE_NAMES,
    LOAN_REQUEST_FEATURES,
    CATEGORICAL_FEATURES_APPLICATION,
    NUMERICAL_FEATURES_APPLICATION,
    FEATURE_NAMES_APPLICATION,
    CATEGORICAL_FEATURES_APPLICANT_ONLY,
    NUMERICAL_FEATURES_APPLICANT_ONLY,
    FEATURE_NAMES_APPLICANT_ONLY,
    CLASSIFICATION_TARGET,
    REGRESSION_TARGET,
    GenderStandardizer,
    get_classification_features,
    get_regression_features,
    build_preprocessor,
    build_classification_pipeline,
    build_regression_pipeline
)

__all__ = [
    "CATEGORICAL_FEATURES",
    "NUMERICAL_FEATURES",
    "DROPPED_FEATURES",
    "FEATURE_NAMES",
    "LOAN_REQUEST_FEATURES",
    "CATEGORICAL_FEATURES_APPLICATION",
    "NUMERICAL_FEATURES_APPLICATION",
    "FEATURE_NAMES_APPLICATION",
    "CATEGORICAL_FEATURES_APPLICANT_ONLY",
    "NUMERICAL_FEATURES_APPLICANT_ONLY",
    "FEATURE_NAMES_APPLICANT_ONLY",
    "CLASSIFICATION_TARGET",
    "REGRESSION_TARGET",
    "GenderStandardizer",
    "get_classification_features",
    "get_regression_features",
    "build_preprocessor",
    "build_classification_pipeline",
    "build_regression_pipeline"
]
