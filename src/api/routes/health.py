"""Health check route."""

from fastapi import APIRouter

from src.core.settings import settings


router = APIRouter(tags=["Health"])


@router.get("/health")
async def health() -> dict[str, str]:
    """Return the API health status."""
    return {
        "status": "API em execução",
        "service": settings.PROJECT_NAME,
    }
