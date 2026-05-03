"""
/api/v1/jobs

POST   /jobs               → create extraction job (returns job_id immediately)
GET    /jobs/{job_id}      → job status + progress
GET    /jobs/{job_id}/results → paginated structured restaurants
DELETE /jobs/{job_id}      → cancel / remove
"""
from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status

from app.config import Settings, get_settings
from app.models.restaurant import (
    ExtractionJobRequest,
    ExtractionJobResponse,
    JobStatus,
    JobStatusResponse,
    Restaurant,
    RestaurantListResponse,
)
from app.services.extractor import RestaurantExtractor
from app.services.job_store import JobStore, JobRecord
from app.services.llm_client import LLMClient

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/jobs", tags=["jobs"])


# ---------------------------------------------------------------------------
# Dependency providers
# ---------------------------------------------------------------------------

def get_llm_client(settings: Annotated[Settings, Depends(get_settings)]) -> LLMClient:
    return LLMClient(settings)


def get_extractor(llm: Annotated[LLMClient, Depends(get_llm_client)]) -> RestaurantExtractor:
    return RestaurantExtractor(llm)


# Singleton job store shared across requests (attached to app.state in main.py)
def get_job_store() -> JobStore:
    # Overridden in app startup via dependency_overrides
    raise NotImplementedError  # pragma: no cover


# ---------------------------------------------------------------------------
# Background task: extraction pipeline
# ---------------------------------------------------------------------------

async def _run_extraction(
    record: JobRecord,
    extractor: RestaurantExtractor,
    store: JobStore,
    concurrency: int,
) -> None:
    try:
        paragraphs = await extractor.load_and_split(record.source_file)
        store.update(record.job_id, status=JobStatus.RUNNING, total=len(paragraphs))

        semaphore = asyncio.Semaphore(concurrency)
        results: list[Optional[Restaurant]] = [None] * len(paragraphs)
        failed: list[int] = []

        async def _process(idx: int, paragraph: str) -> None:
            async with semaphore:
                try:
                    restaurant = await extractor.extract(paragraph)
                    # Assign deterministic itemId matching the notebook convention
                    restaurant.item_id = 1_000_001 + idx
                    results[idx] = restaurant
                except Exception as exc:
                    logger.error(
                        "Failed to extract restaurant",
                        extra={"index": idx, "error": str(exc)},
                    )
                    failed.append(idx)
                finally:
                    processed = sum(1 for r in results if r is not None) + len(failed)
                    store.update(record.job_id, processed=processed, failed_indices=sorted(failed))

        await asyncio.gather(*[_process(i, p) for i, p in enumerate(paragraphs)])

        successful = [r for r in results if r is not None]
        store.update(
            record.job_id,
            status=JobStatus.COMPLETED,
            results=successful,
            failed_indices=sorted(failed),
        )
        logger.info(
            "Extraction job completed",
            extra={
                "job_id": record.job_id,
                "total": len(paragraphs),
                "successful": len(successful),
                "failed": len(failed),
            },
        )
    except Exception as exc:
        logger.exception("Extraction job crashed", extra={"job_id": record.job_id})
        store.update(record.job_id, status=JobStatus.FAILED, error=str(exc))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("", response_model=ExtractionJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_job(
    body: ExtractionJobRequest,
    background_tasks: BackgroundTasks,
    settings: Annotated[Settings, Depends(get_settings)],
    extractor: Annotated[RestaurantExtractor, Depends(get_extractor)],
    store: Annotated[JobStore, Depends(get_job_store)],
) -> ExtractionJobResponse:
    """
    Enqueue an extraction job against `source_file`.
    Returns a `job_id` immediately; poll GET /jobs/{job_id} for progress.
    """
    record = store.create(source_file=body.source_file)
    background_tasks.add_task(
        _run_extraction, record, extractor, store, settings.llm_concurrency
    )
    logger.info("Job created", extra={"job_id": record.job_id, "file": body.source_file})
    return ExtractionJobResponse(
        job_id=record.job_id,
        status=JobStatus.PENDING,
        message="Job accepted. Poll GET /v1/jobs/{job_id} for progress.",
    )


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    store: Annotated[JobStore, Depends(get_job_store)],
) -> JobStatusResponse:
    record = _get_or_404(store, job_id)
    return JobStatusResponse(
        job_id=record.job_id,
        status=record.status,
        total=record.total,
        processed=record.processed,
        failed_indices=record.failed_indices,
        error=record.error,
    )


@router.get("/{job_id}/results", response_model=RestaurantListResponse)
async def get_job_results(
    job_id: str,
    store: Annotated[JobStore, Depends(get_job_store)],
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
) -> RestaurantListResponse:
    record = _get_or_404(store, job_id)

    if record.status not in (JobStatus.COMPLETED, JobStatus.RUNNING):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Results not available yet. Job status: {record.status}",
        )

    page = record.results[offset : offset + limit]
    return RestaurantListResponse(
        job_id=job_id,
        count=len(record.results),
        restaurants=page,
    )


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_job(
    job_id: str,
    store: Annotated[JobStore, Depends(get_job_store)],
) -> None:
    _get_or_404(store, job_id)
    with store._lock:
        store._jobs.pop(job_id, None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_or_404(store: JobStore, job_id: str) -> JobRecord:
    record = store.get(job_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return record
