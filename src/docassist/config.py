from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, read from DOCASSIST_* environment variables and `.env`."""

    model_config = SettingsConfigDict(env_prefix="DOCASSIST_", env_file=".env", extra="ignore")

    # Needed only by the answering route; the API starts without it.
    anthropic_api_key: SecretStr | None = None
    # Always passed to the client explicitly, so a globally exported
    # ANTHROPIC_BASE_URL (e.g. a local proxy) is never picked up.
    anthropic_base_url: str = "https://api.anthropic.com"
    model: str = "claude-opus-5-5"

    database_url: str = "postgresql+asyncpg://docassist:docassist@localhost:5432/docassist"
    db_connect_timeout_s: float = 5.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
