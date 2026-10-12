import json

import joblib
import pytest

import inference
from common import FEATURES, TARGET, build_model, clean

CAR = {"displacement": 140, "horsepower": 90, "weight": 2264,
       "acceleration": 15.5, "model_year": 76, "origin": "usa"}


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory, raw_mpg):
    data = clean(raw_mpg)
    directory = tmp_path_factory.mktemp("model")
    joblib.dump(build_model().fit(data[FEATURES], data[TARGET]), directory / "model.joblib")
    return str(directory)


def invoke(model_dir, body, content_type="application/json"):
    model = inference.model_fn(model_dir)
    data = inference.input_fn(body, content_type)
    return inference.output_fn(inference.predict_fn(data, model), "application/json")


def test_single_object_round_trip(model_dir):
    body, content_type = invoke(model_dir, json.dumps(CAR))

    predictions = json.loads(body)["predicted_mpg"]
    assert content_type == "application/json"
    assert len(predictions) == 1
    assert 20 < predictions[0] < 35


def test_list_of_objects_returns_one_prediction_each(model_dir):
    body, _ = invoke(model_dir, json.dumps([CAR, {**CAR, "weight": 4500, "horsepower": 200}]))

    light, heavy = json.loads(body)["predicted_mpg"]
    assert heavy < light


def test_rejects_non_json_content_type():
    with pytest.raises(ValueError, match="content type"):
        inference.input_fn("1,2,3", "text/csv")


def test_rejects_missing_feature():
    payload = {k: v for k, v in CAR.items() if k != "origin"}

    with pytest.raises(ValueError, match="origin"):
        inference.input_fn(json.dumps(payload), "application/json")


def test_rejects_empty_list():
    with pytest.raises(ValueError):
        inference.input_fn("[]", "application/json")
