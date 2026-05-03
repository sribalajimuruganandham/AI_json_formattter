from __future__ import annotations

import uuid
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Domain model — mirrors the notebook's Restaurant schema exactly
# ---------------------------------------------------------------------------

class Restaurant(BaseModel):
    item_id: Optional[int] = Field(None, alias="itemId")
    name: str
    location: str
    type: str
    food_style: str
    rating: Optional[float] = None
    price_range: Optional[int] = None       # 1=budget … 4=fine dining
    signatures: List[str] = Field(default_factory=list)
    vibe: Optional[str] = None
    environment: str
    shortcomings: List[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Job tracking schemas
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ExtractionJobRequest(BaseModel):
    """POST /v1/jobs body."""
    source_file: str = Field(
        ...,
        description="Server-side path to the raw .txt file",
        examples=["California-Culinary-Map.txt"],
    )


class ExtractionJobResponse(BaseModel):
    """Returned immediately after job creation."""
    job_id: str
    status: JobStatus
    message: str


class JobStatusResponse(BaseModel):
    """GET /v1/jobs/{job_id}"""
    job_id: str
    status: JobStatus
    total: int = 0
    processed: int = 0
    failed_indices: List[int] = Field(default_factory=list)
    error: Optional[str] = None


class RestaurantListResponse(BaseModel):
    """GET /v1/jobs/{job_id}/results"""
    job_id: str
    count: int
    restaurants: List[Restaurant]
