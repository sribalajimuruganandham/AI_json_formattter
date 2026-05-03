"""
Thin async-compatible wrapper around ibm_watsonx_ai ModelInference.

ibm_watsonx_ai is synchronous, so we run calls in a thread-pool via
asyncio.to_thread so they don't block the FastAPI event loop.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import Settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Stateless WatsonX chat client, one instance per app lifetime."""

    def __init__(self, settings: Settings) -> None:
        credentials = Credentials(url=settings.watsonx_url)

        params: dict[str, Any] = {
            GenParams.TEMPERATURE: settings.llm_temperature,
            GenParams.MAX_NEW_TOKENS: settings.llm_max_new_tokens,
        }

        self._model = ModelInference(
            credentials=credentials,
            project_id=settings.watsonx_project_id,
            model_id=settings.watsonx_model_id,
            params=params,
        )
        self._max_retries = settings.llm_max_retries
        self._retry_wait = settings.llm_retry_wait_seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def chat(self, system_msg: str, user_msg: str) -> str:
        """Async chat — runs the sync SDK call in a thread pool."""
        return await asyncio.to_thread(self._sync_chat, system_msg, user_msg)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sync_chat(self, system_msg: str, user_msg: str) -> str:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        response = self._chat_with_retry(messages)
        return response["choices"][0]["message"]["content"]

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _chat_with_retry(self, messages: list[dict]) -> dict:
        try:
            return self._model.chat(messages=messages)
        except Exception as exc:
            logger.warning("WatsonX call failed, will retry", extra={"error": str(exc)})
            raise
