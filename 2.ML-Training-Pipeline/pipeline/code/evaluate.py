"""Pipeline step 3: score the trained model on the fixed holdout set.

Writes evaluation.json, which the pipeline reads (via a PropertyFile) to decide
whether the model passes the thresholds, and which the Model Registry shows as
the model version's metrics.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import tarfile
import tempfile

import joblib
import pandas as pd

from common import FEATURES, TARGET, clean, prepare_features, regression_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_model(model_dir: str):
    """Training jobs emit model.tar.gz; unpack it before loading the joblib inside."""
    archive = os.path.join(model_dir, "model.tar.gz")
    extract_dir = tempfile.mkdtemp()
    with tarfile.open(archive) as tar:
        tar.extractall(extract_dir)
    return joblib.load(os.path.join(extract_dir, "model.joblib"))


def main(model_dir: str, test_dir: str, output_dir: str) -> dict:
    model = load_model(model_dir)

    files = sorted(glob.glob(os.path.join(test_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No test CSV found under {test_dir}")
    test = clean(pd.concat([pd.read_csv(f) for f in files], ignore_index=True))

    predictions = model.predict(prepare_features(test[FEATURES]))
    report = regression_metrics(test[TARGET], predictions)
    logger.info("Evaluated on %d rows: %s", len(test), report)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "evaluation.json"), "w") as f:
        json.dump(report, f)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/opt/ml/processing/model")
    parser.add_argument("--test-dir", default="/opt/ml/processing/test")
    parser.add_argument("--output-dir", default="/opt/ml/processing/evaluation")
    args = parser.parse_args()
    main(args.model_dir, args.test_dir, args.output_dir)
