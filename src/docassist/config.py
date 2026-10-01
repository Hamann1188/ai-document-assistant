from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# The database column has a fixed dimension, so the model is part of the schema.
# Changing it needs a migration and re-ingestion (see docs/ARCHITECTURE.md, ADR-3).
EMBEDDING_MODEL = "google/embeddinggemma-300m"


class Settings(BaseSettings):
    """Application settings, read from DOCASSIST_* environment variables and `.env`."""

    model_config = SettingsConfigDict(env_prefix="DOCASSIST_", env_file=".env", extra="ignore")

    # Needed only by the answering route; the API starts without it.
    anthropic_api_key: SecretStr | None = None
    # Always passed to the client explicitly, so a globally exported
    # ANTHROPIC_BASE_URL (e.g. a local proxy) is never picked up.
    anthropic_base_url: str = "https://api.anthropic.com"
    model: str = "claude-opus-5-5"
    answer_effort: str = "medium"  # low | medium | high | xhigh | max
    answer_max_tokens: int = 8000  # covers adaptive thinking plus the answer
    retrieval_k: int = 8
    # Server-side retry on another model when the requested model declines (beta, ADR-7).
    refusal_fallback: bool = True
    anthropic_timeout_s: float = 120.0

    database_url: str = "postgresql+asyncpg://docassist:docassist@localhost:5432/docassist"
    db_connect_timeout_s: float = 5.0

    upload_dir: Path = Path("uploads")
    max_upload_mb: int = 20
    max_pages: int = 300
    chunk_max_words: int = 120
    chunk_overlap_words: int = 20
    # Development: models download into this cache (None: fastembed's default).
    embedding_cache_dir: Path | None = None
    # Docker: a plain directory with the model files, baked into the image.
    embedding_model_path: Path | None = None
    # Load the model at startup: a broken model stops the app instead of failing uploads.
    embedding_warmup: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
