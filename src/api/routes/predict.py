"""Prediction route skeleton."""

from fastapi import APIRouter

from src.schemas.predict_schema import PredictRequest, PredictResponse

router = APIRouter(tags=["Predictions"])


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Run model inference",
    description="Recebe as features e retorna a predição do modelo",
)
async def predict(payload: PredictRequest) -> PredictResponse:
    """Process one customer after the inference implementation is added."""
    pass
