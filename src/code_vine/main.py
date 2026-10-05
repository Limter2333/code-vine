from fastapi import FastAPI

from code_vine.api.routes import health
from code_vine.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    docs_url="/docs" if settings.debug else None,
)

app.include_router(health.router, tags=["health"])


@app.get("/", tags=["root"])
async def root() -> dict[str, str]:
    """根路径 - 显示可用端点"""
    return {
        "message": "Welcome to code-vine",
        "health": "/health",
        "docs": "/docs",
    }
