"""Runs preprocess -> train -> evaluate locally, the way the pipeline chains them."""
import json
import shutil
import tarfile

import numpy as np
import pandas as pd
import pytest

import evaluate
import preprocess
import train
from common import FEATURES, REQUIRED_COLUMNS, clean, prepare_features


def test_clean_drops_name_and_incomplete_rows(raw_mpg):
    cleaned = clean(raw_mpg)

    assert list(cleaned.columns) == REQUIRED_COLUMNS
    assert len(cleaned) == 392  # 6 cars have no horsepower
    assert not cleaned.isna().any().any()
    assert cleaned["model_year"].dtype.kind == "i"


def test_clean_rejects_missing_columns(raw_mpg):
    with pytest.raises(ValueError, match="weight"):
        clean(raw_mpg.drop(columns=["weight"]))


def test_clean_drops_rows_with_missing_origin(raw_mpg):
    frame = raw_mpg.head(5).copy()
    frame.loc[0, "origin"] = np.nan

    assert len(clean(frame)) == 4


def test_prepare_features_keeps_model_year_integer():
    frame = pd.DataFrame([{"displacement": 140, "horsepower": 90, "weight": 2264,
                           "acceleration": 15.5, "model_year": 76.0, "origin": "usa"}])

    assert prepare_features(frame)["model_year"].dtype.kind == "i"


def test_preprocess_rejects_too_few_rows(tmp_path, raw_mpg):
    (tmp_path / "in").mkdir()
    raw_mpg.head(10).to_csv(tmp_path / "in" / "small.csv", index=False)

    with pytest.raises(ValueError, match="usable rows"):
        preprocess.main(str(tmp_path / "in"), str(tmp_path / "out"))


@pytest.fixture(scope="module")
def evaluation(tmp_path_factory, holdout_split):
    train_csv, test_csv = holdout_split
    work = tmp_path_factory.mktemp("pipeline")
    for name in ("input", "test", "model_artifacts"):
        (work / name).mkdir()
    shutil.copy(train_csv, work / "input")
    shutil.copy(test_csv, work / "test")

    preprocess.main(str(work / "input"), str(work / "train"))
    model_path = train.main(str(work / "train"), str(work / "model"))
    # SageMaker hands the evaluation step the training job's model.tar.gz
    with tarfile.open(work / "model_artifacts" / "model.tar.gz", "w:gz") as tar:
        tar.add(model_path, arcname="model.joblib")

    report = evaluate.main(str(work / "model_artifacts"), str(work / "test"), str(work / "evaluation"))
    return work, report


def test_evaluate_writes_registry_metrics_file(evaluation):
    work, report = evaluation

    written = json.loads((work / "evaluation" / "evaluation.json").read_text())
    assert written == report
    assert set(written["regression_metrics"]) == {"r2", "rmse", "mae"}


def test_model_matches_notebook_scores_and_passes_default_gate(evaluation):
    _, report = evaluation
    metrics = report["regression_metrics"]

    # Notebook (same split): R2 0.827, RMSE 2.95, MAE 2.24
    assert metrics["r2"]["value"] == pytest.approx(0.827, abs=0.005)
    assert metrics["rmse"]["value"] == pytest.approx(2.95, abs=0.02)
    assert metrics["r2"]["value"] >= 0.80 and metrics["rmse"]["value"] <= 3.5


def test_trained_model_predicts_from_feature_columns(evaluation, raw_mpg):
    import joblib

    work, _ = evaluation
    model = joblib.load(work / "model" / "model.joblib")
    prediction = model.predict(prepare_features(clean(raw_mpg).head(3)[FEATURES]))

    assert prediction.shape == (3,)
