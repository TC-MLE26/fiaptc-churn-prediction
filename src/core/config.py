"""FastAPI application factory."""

import logging

from fastapi import FastAPI

from src.core.logging_config import setup_logging
from src.core.settings import settings

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    setup_logging()

    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=settings.PROJECT_DESCRIPTION,
        debug=settings.DEBUG,
        docs_url="/docs" if settings.DOCS_ENABLED else None,
        redoc_url="/redoc" if settings.DOCS_ENABLED else None,
    )

    logger.info(
        "FastAPI application created: %s v%s",
        settings.PROJECT_NAME,
        settings.VERSION,
    )

    return app
