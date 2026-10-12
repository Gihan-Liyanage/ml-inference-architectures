"""Compiles the pipeline definition offline (moto fakes the S3 code uploads)."""
import json

import boto3
import pytest
from moto import mock_aws

BUCKET = "codewave-test-ml-training"
ROLE = "arn:aws:iam::123456789012:role/sagemaker-test"


@pytest.fixture(scope="module")
def definition():
    from sagemaker.workflow.pipeline_context import PipelineSession

    from pipeline.build_pipeline import build_pipeline

    with mock_aws():
        boto_session = boto3.Session(region_name="eu-west-1")
        boto_session.client("s3").create_bucket(
            Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": "eu-west-1"}
        )
        session = PipelineSession(boto_session=boto_session, default_bucket=BUCKET, default_bucket_prefix="sagemaker")
        yield json.loads(build_pipeline("test", BUCKET, ROLE, session).definition())


def _steps(definition):
    """Flatten the top-level steps and the condition's branches into {name: step}."""
    steps = {}
    for step in definition["Steps"]:
        steps[step["Name"]] = step
        for branch in ("IfSteps", "ElseSteps"):
            for inner in step.get("Arguments", {}).get(branch, []):
                steps[inner["Name"]] = inner
    return steps


def test_parameters(definition):
    params = {p["Name"]: p.get("DefaultValue") for p in definition["Parameters"]}

    assert params["InputDataUri"] == f"s3://{BUCKET}/data/train/train.csv"
    assert params["TestDataUri"] == f"s3://{BUCKET}/data/test/mpg_test.csv"
    assert params["R2Threshold"] == 0.80
    assert params["RmseThreshold"] == 3.5
    assert params["ModelApprovalStatus"] == "PendingManualApproval"


def test_step_graph(definition):
    steps = _steps(definition)

    assert [s["Name"] for s in definition["Steps"]] == ["Preprocess", "Train", "Evaluate", "CheckMetrics"]
    assert steps["MetricsBelowThreshold"]["Type"] == "Fail"
    assert [s["Name"] for s in steps["CheckMetrics"]["Arguments"]["ElseSteps"]] == ["MetricsBelowThreshold"]
    # ModelStep expands register into a repack step plus the registration itself
    assert any(s["Type"] == "RegisterModel" for s in steps["CheckMetrics"]["Arguments"]["IfSteps"])


def test_condition_reads_both_metrics_from_evaluation_report(definition):
    conditions = _steps(definition)["CheckMetrics"]["Arguments"]["Conditions"]

    assert [c["Type"] for c in conditions] == ["GreaterThanOrEqualTo", "LessThanOrEqualTo"]
    paths = [c["LeftValue"]["Std:JsonGet"]["Path"] for c in conditions]
    assert paths == ["regression_metrics.r2.value", "regression_metrics.rmse.value"]


def test_registers_into_stage_group_pending_approval(definition):
    register = next(s for s in _steps(definition).values() if s["Type"] == "RegisterModel")
    args = register["Arguments"]

    assert args["ModelPackageGroupName"] == "mpg-linear-regression-test"
    assert args["ModelApprovalStatus"] == {"Get": "Parameters.ModelApprovalStatus"}
    assert args["ModelMetrics"]["ModelQuality"]["Statistics"]["ContentType"] == "application/json"
