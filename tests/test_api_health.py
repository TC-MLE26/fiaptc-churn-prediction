"""Tests for GET /health."""

import main
from src.core.settings import settings


def test_health_returns_200_with_expected_body(client):
    """/health responds with the API status and service name."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "API em execução",
        "service": settings.PROJECT_NAME,
    }
    assert main.app.title == "fiaptc-churn-prediction"
    assert main.app.description == "fiaptc-churn-prediction"
