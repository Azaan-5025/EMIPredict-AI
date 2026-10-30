"""
Phase 5: Environment & Dependency Verification Smoke Tests
Tests all installed packages, imports, and minimal model fits in .venv.
"""

import sys
import numpy as np
import pandas as pd
import pytest

def test_python_version():
    assert sys.version_info >= (3, 12), f"Expected Python 3.12+, got {sys.version}"

def test_core_data_libraries():
    import matplotlib
    import seaborn
    assert np.__version__ is not None
    assert pd.__version__ is not None
    assert matplotlib.__version__ is not None
    assert seaborn.__version__ is not None

def test_scikit_learn_classification_and_regression():
    import sklearn
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, mean_squared_error
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    # Minimal classification test
    X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [5.0, 6.0]])
    y_cls = np.array([0, 0, 1, 1])

    clf_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression())
    ])
    clf_pipe.fit(X, y_cls)
    preds = clf_pipe.predict(X)
    assert len(preds) == 4
    assert accuracy_score(y_cls, preds) >= 0.5

    # Minimal Random Forest Classifier test
    rf_clf = RandomForestClassifier(n_estimators=5, random_state=42)
    rf_clf.fit(X, y_cls)
    assert len(rf_clf.predict(X)) == 4

    # Minimal Linear Regression test
    y_reg = np.array([10.0, 20.0, 30.0, 50.0])
    reg = LinearRegression()
    reg.fit(X, y_reg)
    reg_preds = reg.predict(X)
    assert len(reg_preds) == 4
    assert mean_squared_error(y_reg, reg_preds) >= 0

    # Minimal Random Forest Regressor test
    rf_reg = RandomForestRegressor(n_estimators=5, random_state=42)
    rf_reg.fit(X, y_reg)
    assert len(rf_reg.predict(X)) == 4

def test_xgboost_classification_and_regression():
    import xgboost as xgb
    from xgboost import XGBClassifier, XGBRegressor

    X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [5.0, 6.0]])
    y_cls = np.array([0, 0, 1, 1])
    y_reg = np.array([10.0, 20.0, 30.0, 50.0])

    xgb_clf = XGBClassifier(n_estimators=5, max_depth=2, random_state=42, eval_metric="logloss")
    xgb_clf.fit(X, y_cls)
    preds = xgb_clf.predict(X)
    assert len(preds) == 4

    xgb_reg = XGBRegressor(n_estimators=5, max_depth=2, random_state=42)
    xgb_reg.fit(X, y_reg)
    reg_preds = xgb_reg.predict(X)
    assert len(reg_preds) == 4

def test_mlflow():
    import mlflow
    assert mlflow.__version__ is not None
    # Verify local tracking URI setup without launching a server
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = mlflow.tracking.MlflowClient()
    assert client is not None

def test_streamlit():
    import streamlit as st
    assert st.__version__ is not None

if __name__ == "__main__":
    pytest.main(["-v", __file__])
