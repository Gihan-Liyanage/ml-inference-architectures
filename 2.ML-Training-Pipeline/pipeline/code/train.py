"""Pipeline step 2: fit the regression pipeline (SageMaker SKLearn script mode)."""
from __future__ import annotations

import argparse
import logging
import os

import joblib
import pandas as pd

from common import FEATURES, TARGET, build_model, prepare_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_FILENAME = "model.joblib"


def main(train_dir: str, model_dir: str) -> str:
    frame = pd.read_csv(os.path.join(train_dir, "train.csv"))
    X = prepare_features(frame[FEATURES])
    y = frame[TARGET]

    model = build_model().fit(X, y)
    logger.info("Trained on %d rows", len(frame))

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, MODEL_FILENAME)
    joblib.dump(model, model_path)
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    args, _ = parser.parse_known_args()
    main(args.train_dir, args.model_dir)
