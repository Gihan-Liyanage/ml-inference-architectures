import io
import logging

import boto3
import joblib
from botocore.exceptions import BotoCoreError, ClientError

from src.config import MODEL_BUCKET_NAME, MODEL_OBJECT_KEY
from src.utils.errors import ModelLoadError

logger = logging.getLogger(__name__)

_model = None
_s3_client = None


def get_model():
    """Return the cached model, loading it from S3 on first use in this container."""
    global _model
    if _model is not None:
        return _model

    logger.info(
        "Cold cache miss — loading model from s3://%s/%s",
        MODEL_BUCKET_NAME,
        MODEL_OBJECT_KEY,
    )
    try:
        client = _get_s3_client()
        obj = client.get_object(Bucket=MODEL_BUCKET_NAME, Key=MODEL_OBJECT_KEY)
        buffer = io.BytesIO(obj["Body"].read())
        model = joblib.load(buffer)
    except (ClientError, BotoCoreError) as exc:
        logger.exception("S3 access failed while loading model")
        raise ModelLoadError("Unable to fetch model artifact from S3") from exc
    except Exception as exc:
        logger.exception("Model deserialization failed")
        raise ModelLoadError("Unable to deserialize model artifact") from exc

    _model = model
    logger.info("Model loaded and cached for this execution environment")
    return _model


def _get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3")
    return _s3_client
