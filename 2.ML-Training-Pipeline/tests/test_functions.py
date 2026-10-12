import boto3
import pytest
from botocore.stub import ANY, Stubber

from functions import deploy_model, start_pipeline

S3_EVENT = {
    "detail-type": "Object Created",
    "source": "aws.s3",
    "detail": {
        "bucket": {"name": "codewave-test-ml-training"},
        "object": {"key": "data/train/mpg 2026_10.csv", "etag": "abc", "version-id": "v1", "sequencer": "00A"},
    },
}

APPROVED_EVENT = {
    "detail-type": "SageMaker Model Package State Change",
    "source": "aws.sagemaker",
    "detail": {
        "ModelPackageGroupName": "mpg-linear-regression-test",
        "ModelPackageVersion": 3,
        "ModelPackageArn": "arn:aws:sagemaker:eu-west-1:123456789012:model-package/mpg-linear-regression-test/3",
        "ModelApprovalStatus": "Approved",
    },
}


@pytest.fixture
def sagemaker_stub(monkeypatch):
    client = boto3.client("sagemaker", region_name="eu-west-1")
    with Stubber(client) as stubber:
        monkeypatch.setattr(start_pipeline, "_sagemaker_client", client)
        monkeypatch.setattr(deploy_model, "_sagemaker_client", client)
        yield stubber
        stubber.assert_no_pending_responses()


# --- start_pipeline

def test_start_pipeline_passes_object_uri_and_thresholds(sagemaker_stub):
    sagemaker_stub.add_response(
        "start_pipeline_execution",
        {"PipelineExecutionArn": "arn:aws:sagemaker:eu-west-1:123456789012:pipeline/p/execution/1"},
        {
            "PipelineName": "mpg-training-test",
            "PipelineExecutionDisplayName": "mpg-2026-10",
            "PipelineParameters": [
                {"Name": "InputDataUri", "Value": "s3://codewave-test-ml-training/data/train/mpg 2026_10.csv"},
                {"Name": "R2Threshold", "Value": "0.80"},
                {"Name": "RmseThreshold", "Value": "3.5"},
            ],
            "ClientRequestToken": ANY,
        },
    )

    result = start_pipeline.handler(S3_EVENT, None)

    assert result["inputDataUri"].endswith("mpg 2026_10.csv")


def test_request_token_is_stable_for_retries_and_unique_per_upload():
    obj = S3_EVENT["detail"]["object"]
    token = start_pipeline.request_token("b", obj)

    assert token == start_pipeline.request_token("b", dict(obj))
    assert token != start_pipeline.request_token("b", {**obj, "version-id": "v2"})
    assert 32 <= len(token) <= 128


@pytest.mark.parametrize(
    "key, expected",
    [
        ("data/train/train.csv", "train"),
        ("data/train/2026/__weird name!!.csv", "weird-name"),
        ("data/train/___.csv", "training-run"),
        ("data/train/" + "a" * 200 + ".csv", "a" * 82),
    ],
)
def test_display_name_is_sagemaker_safe(key, expected):
    assert start_pipeline.display_name(key) == expected


# --- deploy_model

def _expect_model_and_config(stubber):
    stubber.add_response(
        "create_model",
        {"ModelArn": "arn:aws:sagemaker:eu-west-1:123456789012:model/m"},
        {
            "ModelName": ANY,
            "ExecutionRoleArn": "arn:aws:iam::123456789012:role/sagemaker-test",
            "Containers": [{"ModelPackageName": APPROVED_EVENT["detail"]["ModelPackageArn"]}],
        },
    )
    stubber.add_response(
        "create_endpoint_config",
        {"EndpointConfigArn": "arn:aws:sagemaker:eu-west-1:123456789012:endpoint-config/c"},
        {
            "EndpointConfigName": ANY,
            "ProductionVariants": [
                {
                    "VariantName": "AllTraffic",
                    "ModelName": ANY,
                    "ServerlessConfig": {"MemorySizeInMB": 2048, "MaxConcurrency": 5},
                }
            ],
        },
    )


def test_deploy_creates_endpoint_when_missing(sagemaker_stub):
    _expect_model_and_config(sagemaker_stub)
    sagemaker_stub.add_client_error(
        "describe_endpoint", service_error_code="ValidationException", service_message="Could not find endpoint"
    )
    sagemaker_stub.add_response(
        "create_endpoint",
        {"EndpointArn": "arn:aws:sagemaker:eu-west-1:123456789012:endpoint/mpg-test-serverless"},
        {"EndpointName": "mpg-test-serverless", "EndpointConfigName": ANY},
    )

    result = deploy_model.handler(APPROVED_EVENT, None)

    assert result["status"] == "created"
    assert result["endpointConfigName"].startswith("mpg-test-serverless-v3-")
    assert len(result["endpointConfigName"]) <= 63


def test_deploy_updates_existing_endpoint(sagemaker_stub):
    _expect_model_and_config(sagemaker_stub)
    sagemaker_stub.add_response(
        "describe_endpoint",
        {
            "EndpointName": "mpg-test-serverless",
            "EndpointArn": "arn:aws:sagemaker:eu-west-1:123456789012:endpoint/mpg-test-serverless",
            "EndpointStatus": "InService",
            "CreationTime": "2026-10-01T00:00:00Z",
            "LastModifiedTime": "2026-10-01T00:00:00Z",
        },
        {"EndpointName": "mpg-test-serverless"},
    )
    sagemaker_stub.add_response(
        "update_endpoint",
        {"EndpointArn": "arn:aws:sagemaker:eu-west-1:123456789012:endpoint/mpg-test-serverless"},
        {"EndpointName": "mpg-test-serverless", "EndpointConfigName": ANY},
    )

    assert deploy_model.handler(APPROVED_EVENT, None)["status"] == "updated"


@pytest.mark.parametrize(
    "override",
    [{"ModelApprovalStatus": "Rejected"}, {"ModelApprovalStatus": "PendingManualApproval"},
     {"ModelPackageGroupName": "some-other-group"}],
)
def test_deploy_ignores_other_events(sagemaker_stub, override):
    event = {**APPROVED_EVENT, "detail": {**APPROVED_EVENT["detail"], **override}}

    assert deploy_model.handler(event, None) == {"status": "ignored"}
