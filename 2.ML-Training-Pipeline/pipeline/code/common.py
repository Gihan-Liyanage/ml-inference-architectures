"""Model definition shared by every pipeline step and the inference handler.

This is the notebook's "Multiple Linear Regression with Categorical Predictors"
section, extracted so preprocessing, training, evaluation and serving all agree
on the same columns and dtypes. It runs inside the SageMaker SKLearn container,
whose Python is older than the local toolchain, so keep it free of newer syntax.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "mpg"
NUMERICAL_FEATURES = ["displacement", "horsepower", "weight", "acceleration"]
CATEGORICAL_FEATURES = ["model_year", "origin"]
FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = [TARGET] + FEATURES


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the feature columns and coerce them to the dtypes the model was fit on.

    `model_year` must stay an int: the OneHotEncoder learns its categories from the
    training dtype, so a float 76.0 at inference time would be an unknown category.
    """
    missing = [c for c in FEATURES if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    features = frame[FEATURES].copy()
    for column in NUMERICAL_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="coerce").astype(float)
    features["model_year"] = pd.to_numeric(features["model_year"], errors="coerce")
    if features["model_year"].notna().all():
        features["model_year"] = features["model_year"].astype(int)
    # astype(str) would turn a missing origin into the string "nan"; keep it missing.
    origin = features["origin"]
    features["origin"] = origin.where(origin.isna(), origin.astype(str))
    return features


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate a raw Auto MPG frame and return only complete, correctly typed rows."""
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    cleaned = prepare_features(frame)
    cleaned[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    cleaned = cleaned.dropna().reset_index(drop=True)
    cleaned["model_year"] = cleaned["model_year"].astype(int)
    return cleaned[REQUIRED_COLUMNS]


def build_model() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", LinearRegression())])


def regression_metrics(y_true, y_pred) -> dict:
    """Metrics in the shape the SageMaker Model Registry expects for regression models."""
    return {
        "regression_metrics": {
            "r2": {"value": float(r2_score(y_true, y_pred))},
            "rmse": {"value": float(np.sqrt(mean_squared_error(y_true, y_pred)))},
            "mae": {"value": float(mean_absolute_error(y_true, y_pred))},
        }
    }
