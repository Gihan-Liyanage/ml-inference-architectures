"""Pipeline step 1: validate the uploaded training CSV(s) and write a clean training set."""
from __future__ import annotations

import argparse
import glob
import logging
import os

import pandas as pd

from common import clean

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MIN_ROWS = 50


def main(input_dir: str, output_dir: str, min_rows: int = MIN_ROWS) -> str:
    files = sorted(glob.glob(os.path.join(input_dir, "**", "*.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {input_dir}")

    raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    cleaned = clean(raw)
    logger.info("Read %d rows from %d file(s); %d usable after cleaning", len(raw), len(files), len(cleaned))

    if len(cleaned) < min_rows:
        raise ValueError(f"Only {len(cleaned)} usable rows; need at least {min_rows} to train")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "train.csv")
    cleaned.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="/opt/ml/processing/input")
    parser.add_argument("--output-dir", default="/opt/ml/processing/train")
    parser.add_argument("--min-rows", type=int, default=MIN_ROWS)
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.min_rows)
