"""One-off: split the Auto MPG CSV into a training file and the fixed holdout test set.

    poetry run python scripts/split_holdout.py

Uses the notebook's split (25%, random_state=42) on rows with complete values, so
the pipeline's first evaluation is comparable to the notebook's test scores.
Writes data/train.csv and data/mpg_test.csv in the raw format (incl. `name`).
Upload mpg_test.csv once to s3://<bucket>/data/test/; uploading train.csv to
s3://<bucket>/data/train/ starts a pipeline run.
"""
import argparse
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline", "code"))

from common import REQUIRED_COLUMNS  # noqa: E402

DEFAULT_SOURCE = os.path.join(ROOT, "seed-data", "mpg.csv")


def main(source: str, output_dir: str, test_size: float, seed: int) -> None:
    raw = pd.read_csv(source)
    complete = raw.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)
    train, test = train_test_split(complete, test_size=test_size, random_state=seed)

    os.makedirs(output_dir, exist_ok=True)
    train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test.to_csv(os.path.join(output_dir, "mpg_test.csv"), index=False)
    print(f"{len(train)} training rows -> {output_dir}/train.csv")
    print(f"{len(test)} holdout rows  -> {output_dir}/mpg_test.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "data"))
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.source, args.output_dir, args.test_size, args.seed)
