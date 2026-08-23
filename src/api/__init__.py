# src/core/__init__.py
"""
Módulo core com funções de configuração da aplicação...
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

def create_app() -> FastAPI:
    """Cria a aplicação FastAPI"""
    app = FastAPI(
        title="Churn Prediction API",
        description="API para predição de churn de clientes de telecom",
        version="1.0.0"
    )
    return app

def setup_cors(app: FastAPI) -> None:
    """Configura CORS para a aplicação."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )