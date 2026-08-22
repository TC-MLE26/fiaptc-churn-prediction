"""Prediction route skeleton."""

from fastapi import APIRouter
from src.schemas.predict_schema import PredictRequest, PredictResponse
from src.ml.model import prever_churn

router = APIRouter(tags=["Predictions"])

@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Run model inference",
    description="Recebe as features e retorna a predição do modelo",
)
async def predict(payload: PredictRequest) -> PredictResponse:
    dados = payload.model_dump()
    resultado_ml = prever_churn(dados)
    return PredictResponse(**resultado_ml)
