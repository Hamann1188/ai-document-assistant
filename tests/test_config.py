import re
from pathlib import Path

from docassist.config import EMBEDDING_MODEL, Settings
from docassist.db.models import EMBEDDING_DIM

ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_bakes_the_configured_embedding_model():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert re.search(r"^ARG EMBEDDING_MODEL=(\S+)$", dockerfile, re.M)[1] == EMBEDDING_MODEL


def test_latest_migration_matches_embedding_dimension():
    dims = [
        int(dim)
        for path in sorted((ROOT / "alembic" / "versions").glob("*.py"))
        for dim in re.findall(r"^EMBEDDING_DIM = (\d+)$", path.read_text(encoding="utf-8"), re.M)
    ]
    assert dims[-1] == EMBEDDING_DIM


def test_defaults_point_at_public_api_and_opus(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:8787")
    settings = Settings(_env_file=None)
    # The unprefixed variable must not leak into our settings.
    assert settings.anthropic_base_url == "https://api.anthropic.com"
    assert settings.model == "claude-opus-5-5"
    assert settings.anthropic_api_key is None


def test_prefixed_env_overrides(monkeypatch):
    monkeypatch.setenv("DOCASSIST_MODEL", "claude-sonnet-5-5")
    monkeypatch.setenv("DOCASSIST_ANTHROPIC_API_KEY", "sk-test")
    settings = Settings(_env_file=None)
    assert settings.model == "claude-sonnet-5-5"
    assert settings.anthropic_api_key.get_secret_value() == "sk-test"
    assert "sk-test" not in repr(settings)
