"""Deploy an approved model package to the SageMaker Serverless endpoint.

Triggered by EventBridge "SageMaker Model Package State Change" events. The rule
already filters on group + Approved status; the checks below are a second guard.
"""
import logging
import os
import time

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())

MODEL_PACKAGE_GROUP_NAME = os.environ["MODEL_PACKAGE_GROUP_NAME"]
ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]
EXECUTION_ROLE_ARN = os.environ["SAGEMAKER_EXECUTION_ROLE_ARN"]
MEMORY_SIZE_MB = int(os.environ.get("SERVERLESS_MEMORY_MB", "2048"))
MAX_CONCURRENCY = int(os.environ.get("SERVERLESS_MAX_CONCURRENCY", "5"))

_sagemaker_client = None


def handler(event, context):
    detail = event["detail"]
    if detail.get("ModelPackageGroupName") != MODEL_PACKAGE_GROUP_NAME:
        logger.info("Ignoring package from group %s", detail.get("ModelPackageGroupName"))
        return {"status": "ignored"}
    if detail.get("ModelApprovalStatus") != "Approved":
        logger.info("Ignoring package with status %s", detail.get("ModelApprovalStatus"))
        return {"status": "ignored"}

    package_arn = detail["ModelPackageArn"]
    version = detail.get("ModelPackageVersion", "x")
    # Model and endpoint-config names are immutable resources, so each deploy gets
    # fresh ones; the endpoint name stays fixed for callers.
    resource_name = f"{ENDPOINT_NAME}-v{version}-{time.strftime('%Y%m%d%H%M%S', time.gmtime())}"[:63]

    client = _get_client()
    client.create_model(
        ModelName=resource_name,
        ExecutionRoleArn=EXECUTION_ROLE_ARN,
        Containers=[{"ModelPackageName": package_arn}],
    )
    client.create_endpoint_config(
        EndpointConfigName=resource_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": resource_name,
                "ServerlessConfig": {
                    "MemorySizeInMB": MEMORY_SIZE_MB,
                    "MaxConcurrency": MAX_CONCURRENCY,
                },
            }
        ],
    )

    if _endpoint_exists(client):
        client.update_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=resource_name)
        action = "updated"
    else:
        client.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=resource_name)
        action = "created"

    logger.info("Endpoint %s %s with model package %s (config %s)", ENDPOINT_NAME, action, package_arn, resource_name)
    return {"status": action, "endpointName": ENDPOINT_NAME, "endpointConfigName": resource_name}


def _endpoint_exists(client) -> bool:
    try:
        client.describe_endpoint(EndpointName=ENDPOINT_NAME)
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "ValidationException":
            return False
        raise


def _get_client():
    global _sagemaker_client
    if _sagemaker_client is None:
        _sagemaker_client = boto3.client("sagemaker")
    return _sagemaker_client
