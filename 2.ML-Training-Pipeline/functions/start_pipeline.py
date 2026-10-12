"""Start a pipeline execution when a training CSV lands under data/train/.

Fire-and-forget: StartPipelineExecution returns as soon as the run is queued.
The pipeline's own progress and failures are reported via EventBridge → SNS.
"""
import hashlib
import logging
import os
import re

import boto3

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())

PIPELINE_NAME = os.environ["PIPELINE_NAME"]
R2_THRESHOLD = os.environ["R2_THRESHOLD"]
RMSE_THRESHOLD = os.environ["RMSE_THRESHOLD"]

_sagemaker_client = None


def handler(event, context):
    detail = event["detail"]
    bucket = detail["bucket"]["name"]
    obj = detail["object"]
    key = obj["key"]
    input_uri = f"s3://{bucket}/{key}"

    response = _get_client().start_pipeline_execution(
        PipelineName=PIPELINE_NAME,
        PipelineExecutionDisplayName=display_name(key),
        PipelineParameters=[
            {"Name": "InputDataUri", "Value": input_uri},
            {"Name": "R2Threshold", "Value": R2_THRESHOLD},
            {"Name": "RmseThreshold", "Value": RMSE_THRESHOLD},
        ],
        ClientRequestToken=request_token(bucket, obj),
    )
    execution_arn = response["PipelineExecutionArn"]
    logger.info("Started %s for %s", execution_arn, input_uri)
    return {"pipelineExecutionArn": execution_arn, "inputDataUri": input_uri}


def request_token(bucket: str, obj: dict) -> str:
    """Idempotency token: stable across retries of the same S3 event, new for every upload.

    The bucket is versioned, so version-id is unique per upload, even when the same
    file is uploaded twice. sequencer/etag are fallbacks if versioning is ever off.
    """
    upload_id = obj.get("version-id") or obj.get("sequencer") or obj.get("etag", "")
    return hashlib.sha256(f"{bucket}/{obj['key']}/{upload_id}".encode()).hexdigest()


def display_name(key: str) -> str:
    """Execution display names allow [a-zA-Z0-9-] only, max 82 chars."""
    stem = os.path.splitext(os.path.basename(key))[0]
    name = re.sub(r"[^a-zA-Z0-9]+", "-", stem).strip("-")[:82].strip("-")
    return name or "training-run"


def _get_client():
    global _sagemaker_client
    if _sagemaker_client is None:
        _sagemaker_client = boto3.client("sagemaker")
    return _sagemaker_client
