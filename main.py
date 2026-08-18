"""Main application entry point."""

import logging

import uvicorn

from src.api.routes.health import router as health_router
from src.api.routes.predict import router as predict_router
from src.core import create_app, setup_cors
from src.core.settings import settings

logger = logging.getLogger(__name__)

app = create_app()
setup_cors(app)
app.include_router(health_router)
app.include_router(predict_router)

logger.info("Application startup complete")


if __name__ == "__main__":
    logger.info("Starting server on %s:%s", settings.HOST, settings.PORT)
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if not settings.DEBUG else "debug",
    )
