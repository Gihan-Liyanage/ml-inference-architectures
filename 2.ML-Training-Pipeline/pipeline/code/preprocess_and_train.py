"""TEMPORARY pipeline step 1+2: preprocess and train in a single processing job.

Used while the account has no SageMaker training-job quota (see
TRAIN_IN_PROCESSING_JOB in build_pipeline.py). Runs the same preprocess.py and
train.py logic, then packages the model as model.tar.gz, the same layout a
Training job produces, so Evaluate and the endpoint work unchanged.
"""
from __future__ import annotations

import argparse
import os
import tarfile
import tempfile

import preprocess
import train


def main(input_dir: str, train_dir: str, model_dir: str, min_rows: int = preprocess.MIN_ROWS) -> str:
    preprocess.main(input_dir, train_dir, min_rows)
    model_path = train.main(train_dir, tempfile.mkdtemp())

    os.makedirs(model_dir, exist_ok=True)
    archive = os.path.join(model_dir, "model.tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(model_path, arcname=train.MODEL_FILENAME)
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="/opt/ml/processing/input")
    parser.add_argument("--train-dir", default="/opt/ml/processing/train")
    parser.add_argument("--model-dir", default="/opt/ml/processing/model")
    parser.add_argument("--min-rows", type=int, default=preprocess.MIN_ROWS)
    args = parser.parse_args()
    main(args.input_dir, args.train_dir, args.model_dir, args.min_rows)
