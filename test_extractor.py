"""
Unit tests for RestaurantExtractor.
LLM is mocked — no network calls in CI.
"""
from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.models.restaurant import Restaurant
from app.services.extractor import RestaurantExtractor


VALID_JSON = json.dumps({
    "name": "The Gilded Artichoke",
    "location": "Silver Lake",
    "type": "upscale bistro",
    "food_style": "Farm-to-Table Californian",
    "rating": 4.5,
    "price_range": 4,
    "signatures": ["lavender-rubbed roasted chicken", "heirloom tomato tarts"],
    "vibe": "bohemian chic",
    "environment": "high-end greenhouse with reclaimed wood",
    "shortcomings": [],
})

BROKEN_JSON = '{"name": "Broken", "rating": "not-a-float"}'  # rating wrong type

REPAIRED_JSON = json.dumps({
    "name": "Broken",
    "location": "Unknown",
    "type": "unknown",
    "food_style": "unknown",
    "rating": 4.0,
    "price_range": 2,
    "signatures": [],
    "vibe": None,
    "environment": "unknown",
    "shortcomings": [],
})


def _mock_llm(first_response: str, repair_response: str | None = None) -> MagicMock:
    llm = MagicMock()
    if repair_response is None:
        llm.chat = AsyncMock(return_value=first_response)
    else:
        llm.chat = AsyncMock(side_effect=[first_response, repair_response])
    return llm


@pytest.mark.asyncio
async def test_extract_valid_response():
    extractor = RestaurantExtractor(_mock_llm(VALID_JSON))
    restaurant = await extractor.extract("any paragraph")
    assert isinstance(restaurant, Restaurant)
    assert restaurant.name == "The Gilded Artichoke"
    assert restaurant.rating == 4.5
    assert restaurant.price_range == 4


@pytest.mark.asyncio
async def test_extract_triggers_repair_on_invalid_json():
    extractor = RestaurantExtractor(_mock_llm(BROKEN_JSON, REPAIRED_JSON))
    restaurant = await extractor.extract("any paragraph")
    assert isinstance(restaurant, Restaurant)
    assert restaurant.name == "Broken"
    assert isinstance(restaurant.rating, float)


@pytest.mark.asyncio
async def test_extract_raises_after_max_repair_attempts():
    # LLM always returns broken JSON
    llm = MagicMock()
    llm.chat = AsyncMock(return_value=BROKEN_JSON)
    extractor = RestaurantExtractor(llm)

    with pytest.raises(Exception):
        await extractor.extract("bad paragraph")


@pytest.mark.asyncio
async def test_load_and_split_missing_file():
    extractor = RestaurantExtractor(MagicMock())
    with pytest.raises(FileNotFoundError):
        await extractor.load_and_split("/nonexistent/path.txt")
