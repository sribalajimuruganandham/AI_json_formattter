"""
JobStore
--------
Thread-safe in-memory job registry.

For production, replace the dict backend with Redis + JSON serialisation
or a SQLAlchemy model. The interface stays identical.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.models.restaurant import JobStatus, Restaurant


@dataclass
class JobRecord:
    job_id: str
    source_file: str
    status: JobStatus = JobStatus.PENDING
    total: int = 0
    processed: int = 0
    failed_indices: List[int] = field(default_factory=list)
    error: Optional[str] = None
    results: List[Restaurant] = field(default_factory=list)


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: Dict[str, JobRecord] = {}

    def create(self, source_file: str) -> JobRecord:
        job_id = str(uuid.uuid4())
        record = JobRecord(job_id=job_id, source_file=source_file)
        with self._lock:
            self._jobs[job_id] = record
        return record

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **kwargs) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            for key, value in kwargs.items():
                setattr(record, key, value)
