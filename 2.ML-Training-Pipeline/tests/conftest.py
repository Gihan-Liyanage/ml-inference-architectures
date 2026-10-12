import os
from pathlib import Path

import pandas as pd
import pytest

# The Lambda modules read these eagerly at import time; set test defaults before
# any functions.* module is imported by the test collection process.
os.environ.setdefault("PIPELINE_NAME", "mpg-training-test")
os.environ.setdefault("R2_THRESHOLD", "0.80")
os.environ.setdefault("RMSE_THRESHOLD", "3.5")
os.environ.setdefault("MODEL_PACKAGE_GROUP_NAME", "mpg-linear-regression-test")
os.environ.setdefault("ENDPOINT_NAME", "mpg-test-serverless")
os.environ.setdefault("SAGEMAKER_EXECUTION_ROLE_ARN", "arn:aws:iam::123456789012:role/sagemaker-test")
os.environ.setdefault("AWS_DEFAULT_REGION", "eu-west-1")
os.environ.setdefault("SAGEMAKER_SUPPRESS_V2_WARNING", "1")

MPG_CSV = Path(__file__).resolve().parents[1] / "seed-data" / "mpg.csv"


@pytest.fixture(scope="session")
def raw_mpg() -> pd.DataFrame:
    return pd.read_csv(MPG_CSV)


@pytest.fixture(scope="session")
def holdout_split(tmp_path_factory, raw_mpg):
    """data/train.csv + data/mpg_test.csv exactly as scripts/split_holdout.py writes them."""
    from scripts.split_holdout import main as split

    out = tmp_path_factory.mktemp("data")
    split(str(MPG_CSV), str(out), test_size=0.25, seed=42)
    return out / "train.csv", out / "mpg_test.csv"
