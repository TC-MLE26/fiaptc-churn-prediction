"""Shared pytest fixtures for the API test suite."""

from collections.abc import Iterator

from fastapi.testclient import TestClient
import pytest

import main


@pytest.fixture()
def client() -> Iterator[TestClient]:
    """Test client for the FastAPI application."""
    with TestClient(main.app) as test_client:
        yield test_client


@pytest.fixture()
def valid_payload() -> dict[str, object]:
    """A known-valid PredictRequest payload (mirrors the schema example)."""
    return {
        "tenure_months": 1,
        "monthly_charges": 29.85,
        "total_charges": 29.85,
        "gender": "Female",
        "senior_citizen": "No",
        "partner": "Yes",
        "dependents": "No",
        "phone_service": "No",
        "multiple_lines": "No phone service",
        "internet_service": "DSL",
        "online_security": "No",
        "online_backup": "Yes",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
    }
