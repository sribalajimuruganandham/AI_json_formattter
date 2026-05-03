"""
RestaurantExtractor
-------------------
Owns the full extraction pipeline:
  raw paragraph → LLM → JSON string → Pydantic validation → (repair loop) → Restaurant
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

from pydantic import ValidationError

from app.models.restaurant import Restaurant
from app.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# One-shot example (paragraph index 1 from the dataset)
# ---------------------------------------------------------------------------
_EXAMPLE_INPUT = """
**Mar de Cortez** is a **casual taqueria** in **Santa Monica** that serves **Baja-style seafood**,
earning a **4.2/5** rating for its beer-battered snapper tacos and zesty octopus ceviche.
The salt-air energy makes it a premier sun-drenched spot for open-air dining near the pier.
Price range: $
""".strip()

_EXAMPLE_OUTPUT = """{
    "name": "Mar de Cortez",
    "location": "Santa Monica",
    "type": "casual taqueria",
    "food_style": "Baja-style seafood",
    "rating": 4.2,
    "price_range": 1,
    "signatures": ["beer-battered snapper tacos", "zesty octopus ceviche"],
    "vibe": "salt-air energy",
    "environment": "a premier sun-drenched spot for open-air dining near the pier.",
    "shortcomings": []
}"""

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM = """
You are a structured data extraction expert specialising in the food and hospitality industry.
Extract key information from the restaurant description and return a single valid JSON object.

Rules:
- Output ONLY raw JSON. No markdown, no code fences, no explanation.
- Every field must be present. Use null for missing text fields, [] for missing arrays, 0 for missing numbers.
- rating → float. price_range → integer ($ = 1, $$ = 2, $$$ = 3, $$$$ = 4).
- signatures → list[str] of standout dishes or drinks.
- shortcomings → list[str] of negatives. Empty list if none.
- Do not invent information not present in the description.
""".strip()

_REPAIR_SYSTEM = """
You are a JSON repair expert. Fix the malformed or schema-mismatched JSON below.
Rules:
- Output ONLY the corrected raw JSON. No explanation, no markdown, no code fences.
- Fix all schema violations listed in the error message.
- Do not add or remove fields. Do not invent data.
- Ensure types match: rating=float, price_range=int, signatures=list, shortcomings=list.
""".strip()


def _extraction_prompt(paragraph: str) -> str:
    return f"""
Extract structured restaurant data from the description and return JSON matching the schema.

Example
-------
Input:
{_EXAMPLE_INPUT}

Output:
{_EXAMPLE_OUTPUT}

-------
Now extract from this description:
Input:
{paragraph}

Output:
""".strip()


def _repair_prompt(candidate: str, error: str) -> str:
    return f"""
The following JSON failed schema validation.

Error:
{error}

Broken JSON:
{candidate}

Return only the fixed JSON:
""".strip()


# ---------------------------------------------------------------------------
# Service class
# ---------------------------------------------------------------------------

class RestaurantExtractor:
    MAX_REPAIR_ATTEMPTS = 3

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def extract(self, paragraph: str) -> Restaurant:
        """Extract a single Restaurant from a raw description paragraph."""
        raw = await self._llm.chat(_EXTRACTION_SYSTEM, _extraction_prompt(paragraph))
        return await self._validate_with_repair(raw)

    async def load_and_split(self, file_path: str) -> List[str]:
        """Read the raw .txt file and return one paragraph per restaurant."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")

        text = path.read_text(encoding="utf-8")
        # First paragraph is the dataset title — drop it
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        return paragraphs[1:]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _validate_with_repair(self, raw: str) -> Restaurant:
        candidate = raw
        for attempt in range(self.MAX_REPAIR_ATTEMPTS):
            try:
                return Restaurant.model_validate_json(candidate)
            except ValidationError as exc:
                if attempt == self.MAX_REPAIR_ATTEMPTS - 1:
                    logger.error(
                        "Extraction failed after max repair attempts",
                        extra={"error": exc.json(), "raw": raw[:200]},
                    )
                    raise

                logger.warning(
                    "Validation failed, invoking repair LLM",
                    extra={"attempt": attempt + 1, "error": str(exc)[:300]},
                )
                candidate = await self._llm.chat(
                    _REPAIR_SYSTEM, _repair_prompt(candidate, str(exc))
                )
        # Unreachable — for type checker
        raise RuntimeError("Repair loop exited without returning")
