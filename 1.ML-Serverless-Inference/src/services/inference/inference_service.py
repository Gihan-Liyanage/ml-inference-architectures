import logging

import pandas as pd

from src.models.predict import PredictRequest, PredictResponse
from src.services.inference.model_loader import get_model
from src.utils.errors import UnknownCategoryError

logger = logging.getLogger(__name__)


def predict(payload: PredictRequest) -> PredictResponse:
    model = get_model()
    df = pd.DataFrame([payload.model_dump()])

    try:
        prediction = model.predict(df)
    except ValueError as exc:
        logger.warning("Model rejected input as an unknown category: %s", exc)
        raise UnknownCategoryError(str(exc)) from exc

    return PredictResponse(predicted_mpg=float(prediction[0]), input=payload)
