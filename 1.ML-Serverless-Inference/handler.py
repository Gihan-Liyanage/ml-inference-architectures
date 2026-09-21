import json
import logging

from pydantic import ValidationError

from src.models.predict import PredictRequest
from src.services.inference.inference_service import predict
from src.utils.errors import AppError
from src.utils.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def run(event, context):
    request_id = getattr(context, "aws_request_id", None)
    try:
        payload = PredictRequest.model_validate_json(event.get("body") or "{}")
        result = predict(payload)
        return _response(200, result.model_dump())

    except ValidationError as exc:
        logger.warning("request_id=%s invalid request: %s", request_id, exc)
        return _response(400, {"error": "invalid_request", "message": str(exc)})

    except AppError as exc:
        logger.error(
            "request_id=%s %s: %s", request_id, type(exc).__name__, exc, exc_info=True
        )
        return _response(exc.status_code, {"error": type(exc).__name__, "message": str(exc)})

    except Exception:
        logger.exception("request_id=%s unhandled error", request_id)
        return _response(500, {"error": "internal_error", "message": "An unexpected error occurred"})


def _response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }
