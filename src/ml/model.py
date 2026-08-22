import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "champion_model.joblib"

_pipeline = None

def carregar_modelo():
    global _pipeline
    if _pipeline is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Modelo não encontrado em: {MODEL_PATH}")
        _pipeline = joblib.load(MODEL_PATH)

def prever_churn(dados_cliente: dict) -> dict:
    carregar_modelo()
    
    df_input = pd.DataFrame([dados_cliente])
    
    probabilidade = _pipeline.predict_proba(df_input)[0][1]
    predicao = _pipeline.predict(df_input)[0]
    
    return {
        "churn_probability": float(probabilidade),
        "churn_prediction": bool(predicao),
        "threshold": 0.5,
        "model_name": "Random Forest",
        "model_version": "v1.0"
    }
