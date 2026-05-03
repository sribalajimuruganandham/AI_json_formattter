from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # WatsonX
    watsonx_url: str = "https://us-south.ml.cloud.ibm.com"
    watsonx_project_id: str = "skills-network"
    watsonx_api_key: str = ""
    watsonx_model_id: str = "ibm/granite-4-h-small"

    # LLM behaviour
    llm_temperature: float = 0.1
    llm_max_new_tokens: int = 512
    llm_concurrency: int = 5          # semaphore: max parallel WatsonX calls
    llm_max_retries: int = 3
    llm_retry_wait_seconds: float = 2.0

    # App
    app_env: str = "production"
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
