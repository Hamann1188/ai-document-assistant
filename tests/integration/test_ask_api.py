"""/ask end to end with a real database, the fake embedder and a scripted Claude."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from docassist.api.main import create_app
from tests.conftest import ev_cite, ev_text, final_message

SAMPLE_DOCS = Path(__file__).resolve().parents[2] / "sample_docs"


@pytest.fixture
def claude(fake_claude_cls):
    return fake_claude_cls(
        events=[ev_text(0, "Saturday: "), ev_text(1, "10:00 - 16:00"), ev_cite(1, 0, "x")],
        message=final_message(),
    )


@pytest.fixture
def app_client(settings, embedder, claude):
    app = create_app(
        settings, embedder_factory=lambda _: embedder, llm_client_factory=lambda _: claude
    )
    with TestClient(app) as client:
        data = (SAMPLE_DOCS / "patient-handbook.pdf").read_bytes()
        client.post("/documents", files={"file": ("patient-handbook.pdf", data, "application/pdf")})
        yield client
    assert claude.closed


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for frame in body.strip().split("\n\n"):
        name, data = frame.split("\n")
        events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events


def test_ask_streams_server_sent_events(app_client, claude):
    response = app_client.post("/ask", json={"question": "When are you open on Saturday?", "k": 3})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["sources", "text", "text", "citation", "done"]
    sources = events[0][1]["sources"]
    assert sources and all(s["filename"] == "patient-handbook.pdf" for s in sources)
    citation = events[3][1]
    assert citation["block"] == 1 and citation["page"] == sources[0]["page"]
    assert events[-1][1]["stop_reason"] == "end_turn"

    # Retrieval used k=3 chunks; the request carries one search_result per page.
    content = claude.requests[0]["messages"][0]["content"]
    assert 1 <= sum(b["type"] == "search_result" for b in content) <= 3


def test_ask_returns_json_when_not_streaming(app_client):
    response = app_client.post("/ask", json={"question": "Saturday hours?", "stream": False})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Saturday: 10:00 - 16:00"
    assert body["blocks"][1]["citations"][0]["filename"] == "patient-handbook.pdf"
    assert body["error"] is None and body["cost_usd"] > 0


@pytest.mark.parametrize(
    "payload", [{"question": "   "}, {"question": ""}, {"question": "x" * 2001}, {"k": 3}]
)
def test_ask_validates_input(app_client, payload):
    assert app_client.post("/ask", json=payload).status_code == 422


def test_ask_without_api_key_streams_an_error(settings, embedder):
    app = create_app(
        settings, embedder_factory=lambda _: embedder, llm_client_factory=lambda _: None
    )
    with TestClient(app) as client:
        events = parse_sse(client.post("/ask", json={"question": "Hi?"}).text)
    assert [name for name, _ in events] == ["sources", "error"]
    assert "DOCASSIST_ANTHROPIC_API_KEY" in events[1][1]["message"]
