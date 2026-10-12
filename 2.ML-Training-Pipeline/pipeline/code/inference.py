"""Serving handlers loaded by the SageMaker SKLearn container on the endpoint.

Request (application/json): one object or a list of objects, e.g.
    {"displacement": 140, "horsepower": 90, "weight": 2264,
     "acceleration": 15.5, "model_year": 76, "origin": "usa"}
Response: {"predicted_mpg": [<float>, ...]} (one value per input object).
"""
from __future__ import annotations

import json
import os

import joblib
import pandas as pd

from common import FEATURES, prepare_features

JSON = "application/json"


def model_fn(model_dir: str):
    return joblib.load(os.path.join(model_dir, "model.joblib"))


def input_fn(request_body, request_content_type: str) -> pd.DataFrame:
    if not request_content_type or not request_content_type.startswith(JSON):
        raise ValueError(f"Unsupported content type {request_content_type!r}; send {JSON}")

    payload = json.loads(request_body)
    records = payload if isinstance(payload, list) else [payload]
    if not records or not all(isinstance(r, dict) for r in records):
        raise ValueError("Body must be a JSON object or a non-empty list of objects")

    frame = pd.DataFrame.from_records(records)
    missing = [c for c in FEATURES if c not in frame.columns or frame[c].isna().any()]
    if missing:
        raise ValueError(f"Missing values for: {missing}")
    return prepare_features(frame)


def predict_fn(input_data: pd.DataFrame, model):
    return model.predict(input_data)


def output_fn(prediction, accept: str):
    body = json.dumps({"predicted_mpg": [float(p) for p in prediction]})
    return body, JSON
