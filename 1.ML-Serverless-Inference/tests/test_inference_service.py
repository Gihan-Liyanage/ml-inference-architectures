from unittest.mock import MagicMock

import numpy as np
import pytest

import src.services.inference.model_loader as model_loader
from src.models.predict import PredictRequest
from src.services.inference.inference_service import predict
from src.utils.errors import UnknownCategoryError


@pytest.fixture(autouse=True)
def _reset_model_cache():
    yield
    model_loader._model = None


def _make_request(**overrides):
    defaults = dict(
        displacement=140,
        horsepower=90,
        weight=2264,
        acceleration=15.5,
        model_year=76,
        origin="usa",
    )
    defaults.update(overrides)
    return PredictRequest(**defaults)


def test_predict_returns_the_models_prediction_as_a_float():
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([26.71])
    model_loader._model = mock_model

    result = predict(_make_request())

    assert result.predicted_mpg == pytest.approx(26.71)
    mock_model.predict.assert_called_once()


def test_predict_builds_a_single_row_dataframe_with_expected_columns():
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([20.0])
    model_loader._model = mock_model

    predict(_make_request())

    called_df = mock_model.predict.call_args[0][0]
    assert list(called_df.columns) == [
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model_year",
        "origin",
    ]
    assert len(called_df) == 1


def test_predict_translates_model_value_error_into_unknown_category_error():
    mock_model = MagicMock()
    mock_model.predict.side_effect = ValueError(
        "Found unknown categories ['germany'] in column 1 during transform"
    )
    model_loader._model = mock_model

    with pytest.raises(UnknownCategoryError):
        predict(_make_request(origin="germany"))
