import re

import pytest
from fastapi.testclient import TestClient

from docassist.api.main import create_app
from docassist.api.web import CONTENT_SECURITY_POLICY, WEB_DIR
from docassist.config import Settings
from docassist.db.models import EMBEDDING_DIM


class _NoDatabase:
    async def dispose(self):
        return None


@pytest.fixture
def client(fake_embedder_cls, tmp_path):
    app = create_app(
        Settings(_env_file=None, upload_dir=tmp_path),
        engine_factory=lambda _: _NoDatabase(),
        embedder_factory=lambda _: fake_embedder_cls(EMBEDDING_DIM),
        llm_client_factory=lambda _: None,
        recover_interrupted=False,
    )
    with TestClient(app) as client:
        yield client


def test_index_is_served_with_a_strict_content_security_policy(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["content-security-policy"] == CONTENT_SECURITY_POLICY
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "<title>AI Document Assistant</title>" in response.text


@pytest.mark.parametrize(
    ("path", "content_type"),
    [
        ("/static/app.js", "javascript"),
        ("/static/styles.css", "text/css"),
        ("/static/favicon.svg", "image/svg+xml"),
    ],
)
def test_static_assets_are_served(client, path, content_type):
    response = client.get(path)
    assert response.status_code == 200
    assert content_type in response.headers["content-type"]
    assert "content-security-policy" in response.headers


def test_api_docs_keep_working_without_the_ui_policy(client):
    response = client.get("/docs")
    assert response.status_code == 200
    assert "content-security-policy" not in response.headers


def test_index_references_only_existing_local_assets():
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    assets = re.findall(r'(?:src|href)="/static/([^"]+)"', html)
    assert assets and all((WEB_DIR / name).is_file() for name in assets)
    # No inline scripts or styles: the CSP would block them.
    assert "<style" not in html
    assert not re.search(r"<script(?![^>]*\bsrc=)", html)
    assert " style=" not in html
