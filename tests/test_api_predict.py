"""Tests for POST /predict."""

from src.schemas.predict_schema import PredictRequest


def test_predict_request_accepts_clean_dataset_features(valid_payload):
    """The request contract accepts the 19 features produced by the notebook."""
    request = PredictRequest.model_validate(valid_payload)

    assert request.model_dump() == valid_payload


def test_predict_missing_required_field_returns_422(client, valid_payload):
    """Dropping a required field triggers request validation."""
    payload = dict(valid_payload)
    del payload["gender"]

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "gender"]


def test_predict_invalid_categorical_value_returns_422(client, valid_payload):
    """A categorical value outside the known domain is rejected."""
    payload = dict(valid_payload)
    payload["contract"] = "Weekly"

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_unknown_field_returns_422(client, valid_payload):
    """Fields outside the request contract are rejected."""
    payload = {**valid_payload, "churn": 1}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_contract_is_exposed_in_openapi(client):
    """OpenAPI keeps the request and response contracts fixed."""
    operation = client.get("/openapi.json").json()["paths"]["/predict"]["post"]

    request_schema = operation["requestBody"]["content"]["application/json"]["schema"]
    response_schema = operation["responses"]["200"]["content"]["application/json"][
        "schema"
    ]

    assert request_schema["$ref"].endswith("/PredictRequest")
    assert response_schema["$ref"].endswith("/PredictResponse")
