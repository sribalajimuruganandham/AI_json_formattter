"""
Application factory.

Wiring order:
  1. configure_logging()                   – JSON structured logs
  2. lifespan()                            – attach shared singletons to app.state
  3. register routers                      – /v1/jobs, /health
  4. dependency_overrides                  – inject app.state.job_store into routes
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import time

from app.api.v1 import jobs as jobs_router
from app.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.services.job_store import JobStore

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan – startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)

    logger.info("Starting restaurant-etl-api", extra={"env": settings.app_env})

    # Shared singletons
    app.state.job_store = JobStore()

    # Wire dependency override so routes receive app.state.job_store
    def _job_store_provider() -> JobStore:
        return app.state.job_store

    app.dependency_overrides[jobs_router.get_job_store] = _job_store_provider

    yield  # ← application runs here

    logger.info("Shutting down restaurant-etl-api")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Restaurant ETL API",
        description=(
            "Async LLM-powered pipeline that converts raw restaurant "
            "descriptions into structured JSON via IBM WatsonX."
        ),
        version="1.0.0",
        docs_url="/docs" if settings.app_env != "production" else None,
        redoc_url="/redoc" if settings.app_env != "production" else None,
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------
    # Middleware: request timing + correlation ID
    # ------------------------------------------------------------------
    @app.middleware("http")
    async def request_timing(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        response.headers["X-Response-Time-Ms"] = str(duration_ms)
        logger.info(
            "HTTP request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        return response

    # ------------------------------------------------------------------
    # Global exception handler
    # ------------------------------------------------------------------
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception", extra={"path": request.url.path})
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    # ------------------------------------------------------------------
    # Health check (no auth required, used by load balancers / k8s)
    # ------------------------------------------------------------------
    @app.get("/health", tags=["ops"], include_in_schema=False)
    async def health() -> dict:
        return {"status": "ok"}

    # ------------------------------------------------------------------
    # Routers
    # ------------------------------------------------------------------
    app.include_router(jobs_router.router, prefix="/v1")

    return app


app = create_app()
