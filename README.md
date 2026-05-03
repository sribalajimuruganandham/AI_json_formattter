# Restaurant ETL API

Async FastAPI service that converts raw restaurant description text into structured JSON via IBM WatsonX (Granite).

## Architecture

```
POST /v1/jobs          ← accepts file path, enqueues background job
GET  /v1/jobs/{id}     ← poll status + progress counter
GET  /v1/jobs/{id}/results  ← paginated structured restaurants
DELETE /v1/jobs/{id}   ← remove job
GET  /health           ← liveness probe
```

**Extraction pipeline per restaurant paragraph:**
```
raw paragraph
    │
    ▼
LLM (WatsonX Granite) ──► JSON string
    │
    ▼
Pydantic validation
    │ fail?
    ▼
Repair LLM (up to 3 attempts)
    │
    ▼
Restaurant model → stored in JobRecord
```

Concurrency is controlled by `LLM_CONCURRENCY` (asyncio.Semaphore) to respect WatsonX rate limits.

## Local setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# fill in WATSONX_PROJECT_ID and WATSONX_API_KEY
```

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

Interactive docs at `http://localhost:8000/docs` (non-production only).

## Usage

```bash
# 1. Submit a job
curl -X POST http://localhost:8000/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"source_file": "California-Culinary-Map.txt"}'

# Response: {"job_id": "uuid", "status": "pending", "message": "..."}

# 2. Poll progress
curl http://localhost:8000/v1/jobs/{job_id}

# Response: {"job_id": "...", "status": "running", "total": 210, "processed": 47, ...}

# 3. Fetch results (paginated)
curl "http://localhost:8000/v1/jobs/{job_id}/results?offset=0&limit=50"
```

## Tests

```bash
pytest -v
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `WATSONX_URL` | `https://us-south.ml.cloud.ibm.com` | WatsonX endpoint |
| `WATSONX_PROJECT_ID` | `skills-network` | Project ID |
| `WATSONX_API_KEY` | — | API key |
| `WATSONX_MODEL_ID` | `ibm/granite-4-h-small` | Model |
| `LLM_CONCURRENCY` | `5` | Max parallel WatsonX calls |
| `LLM_MAX_RETRIES` | `3` | Retries per LLM call |
| `APP_ENV` | `production` | Disables `/docs` in production |
| `LOG_LEVEL` | `INFO` | Logging level |

## Swapping the backend

`JobStore` is intentionally isolated. To replace with Redis:
1. Implement the same `create / get / update` interface against Redis hashes.
2. Replace the `app.state.job_store = JobStore()` line in `main.py`.
No route code changes needed.
