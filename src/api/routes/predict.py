"""Churn prediction endpoint."""

import logging
from functools import lru_cache

import pandas as pd
from fastapi import APIRouter

from src.model import ChurnPredictor
from src.schemas.predict_schema import PredictRequest, PredictResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Predictions"])

# Limiar de decisão aplicado à probabilidade predita (classe churn = 1).
# Nenhum threshold está salvo nos artefatos do modelo, então usa-se 0.5.
DEFAULT_THRESHOLD = 0.5

MODEL_NAME = "champion_model"
MODEL_VERSION = "0.1.0"


@lru_cache(maxsize=1)
def get_predictor() -> ChurnPredictor:
    """Carrega o modelo uma única vez (lazy) e o reutiliza entre requests."""
    return ChurnPredictor()


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Run model inference",
    description="Recebe as features e retorna a predição do modelo",
)
def predict(payload: PredictRequest) -> PredictResponse:
    """Run churn inference for one customer.

    A função é síncrona de propósito: o FastAPI a executa no threadpool,
    evitando que pandas/sklearn bloqueiem o event loop.
    """
    predictor = get_predictor()

    # Converte o payload em um DataFrame de 1 linha, com as colunas na mesma
    # ordem usada no treino (NUMERICAS + CATEGORICAS).
    feature_order = predictor.colunas["NUMERICAS"] + predictor.colunas["CATEGORICAS"]
    features = pd.DataFrame([payload.model_dump()], columns=feature_order)

    probability = float(predictor.model.predict_proba(features)[0, 1])
    prediction = probability >= DEFAULT_THRESHOLD

    logger.info(
        "Prediction made: churn_probability=%.4f churn_prediction=%s",
        probability,
        prediction,
    )

    return PredictResponse(
        churn_probability=probability,
        churn_prediction=prediction,
        threshold=DEFAULT_THRESHOLD,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
    )
