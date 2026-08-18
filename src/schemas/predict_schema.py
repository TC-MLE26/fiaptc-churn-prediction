"""Request and response contracts for the churn prediction endpoint.

Field names match the cleaned dataset produced by the project notebook.
Identifiers and the prediction target are not part of the request.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_EXAMPLE_REQUEST = {
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


class PredictRequest(BaseModel):
    """Features required to predict churn for one customer."""

    tenure_months: int = Field(ge=0)
    monthly_charges: float = Field(ge=0)
    total_charges: float = Field(ge=0)
    gender: Literal["Female", "Male"]
    senior_citizen: Literal["No", "Yes"]
    partner: Literal["No", "Yes"]
    dependents: Literal["No", "Yes"]
    phone_service: Literal["No", "Yes"]
    multiple_lines: Literal["No", "No phone service", "Yes"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    online_security: Literal["No", "No internet service", "Yes"]
    online_backup: Literal["No", "No internet service", "Yes"]
    device_protection: Literal["No", "No internet service", "Yes"]
    tech_support: Literal["No", "No internet service", "Yes"]
    streaming_tv: Literal["No", "No internet service", "Yes"]
    streaming_movies: Literal["No", "No internet service", "Yes"]
    contract: Literal["Month-to-month", "One year", "Two year"]
    paperless_billing: Literal["No", "Yes"]
    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": _EXAMPLE_REQUEST},
    )


class PredictResponse(BaseModel):
    """Response contract for a churn prediction."""

    churn_probability: float = Field(
        ge=0, le=1, description="Predicted probability that the customer churns."
    )
    churn_prediction: bool = Field(
        description="Whether the churn probability reached the decision threshold."
    )
    threshold: float = Field(
        ge=0,
        le=1,
        description="Decision threshold applied to the probability.",
    )
    model_name: str = Field(description="Name of the model that produced the prediction.")
    model_version: str = Field(description="Version of the model artifact.")
