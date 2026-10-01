"""Upload checks that reject a request before it touches the database."""

import pytest
from fastapi.testclient import TestClient

from docassist.api.main import create_app
from docassist.config import Settings
from docassist.db.models import EMBEDDING_DIM
from docassist.ingest.pipeline import clean_filename


class _NoDatabase:
    def connect(self):
        raise AssertionError("validation must not reach the database")

    async def dispose(self):
        return None


@pytest.fixture
def client(fake_embedder_cls, tmp_path):
    settings = Settings(_env_file=None, max_upload_mb=1, upload_dir=tmp_path)
    app = create_app(
        settings,
        engine_factory=lambda _: _NoDatabase(),
        embedder_factory=lambda _: fake_embedder_cls(EMBEDDING_DIM),
        recover_interrupted=False,
    )
    with TestClient(app) as client:
        yield client


def test_broken_embedding_model_stops_startup_when_warmup_is_on(fake_embedder_cls, tmp_path):
    class BrokenModel(fake_embedder_cls):
        def embed_query(self, query):
            raise PermissionError("model.onnx: permission denied")

    app = create_app(
        Settings(_env_file=None, upload_dir=tmp_path, embedding_warmup=True),
        engine_factory=lambda _: _NoDatabase(),
        embedder_factory=lambda _: BrokenModel(EMBEDDING_DIM),
        recover_interrupted=False,
    )
    with pytest.raises(PermissionError), TestClient(app):
        pass


def test_rejects_non_pdf_upload(client):
    response = client.post("/documents", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 415


def test_rejects_upload_over_size_limit(client):
    data = b"%PDF-1.7\n" + b"0" * (1024 * 1024)
    response = client.post("/documents", files={"file": ("big.pdf", data, "application/pdf")})
    assert response.status_code == 413
    assert "1 MB" in response.json()["detail"]


def test_rejects_missing_file(client):
    assert client.post("/documents").status_code == 422


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("report.pdf", "report.pdf"),
        ("C:\\Users\\me\\Desktop\\report.pdf", "report.pdf"),
        ("../../etc/passwd", "passwd"),
        ("bad\x00name\x1f.pdf", "badname.pdf"),
        ("", "document.pdf"),
        (None, "document.pdf"),
        ("a" * 300 + ".pdf", "a" * 255),
    ],
)
def test_clean_filename(raw, expected):
    assert clean_filename(raw) == expected
