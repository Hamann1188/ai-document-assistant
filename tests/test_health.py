from contextlib import asynccontextmanager

from fastapi.testclient import TestClient

from docassist.api.main import create_app
from docassist.config import Settings


class _FakeConnection:
    async def execute(self, statement):
        return None


class _FakeEngine:
    @asynccontextmanager
    async def connect(self):
        yield _FakeConnection()

    async def dispose(self):
        return None


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_healthz_ok_when_database_answers():
    app = create_app(_settings(), engine_factory=lambda _: _FakeEngine(), recover_interrupted=False)
    with TestClient(app) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "db": "ok"}


def test_healthz_503_when_database_unreachable():
    # Nothing listens on port 1, so the real driver fails to connect.
    settings = _settings(
        database_url="postgresql+asyncpg://user:pass@127.0.0.1:1/none",
        db_connect_timeout_s=3,
    )
    with TestClient(create_app(settings)) as client:
        response = client.get("/healthz")
    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "db": "unreachable"}
