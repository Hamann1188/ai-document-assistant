import hashlib
from collections.abc import Sequence
from types import SimpleNamespace

import pytest


class FakeEmbedder:
    """Deterministic stand-in for the ONNX model: same text, same unit vector."""

    def __init__(self, dim: int, fail: bool = False) -> None:
        self.dim = dim
        self.fail = fail
        self.calls = 0

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        raw = [digest[i % len(digest)] - 127.5 for i in range(self.dim)]
        norm = sum(x * x for x in raw) ** 0.5
        return [x / norm for x in raw]

    def embed_passages(self, passages: Sequence[tuple[str, str]]) -> list[list[float]]:
        self.calls += 1
        if self.fail:
            raise RuntimeError("embedding backend crashed")
        return [self._vector(f"{title}|{text}") for title, text in passages]

    def embed_query(self, query: str) -> list[float]:
        return self._vector(query)


@pytest.fixture
def fake_embedder_cls() -> type[FakeEmbedder]:
    return FakeEmbedder


# --- Scripted stand-in for AsyncAnthropic (only the parts the app uses) ---------------


def ev_block_start(index: int, block_type: str, cited: bool = False) -> SimpleNamespace:
    # The API marks a text block that will carry citations with `citations: []`.
    block = SimpleNamespace(type=block_type, citations=[] if cited else None)
    return SimpleNamespace(type="content_block_start", index=index, content_block=block)


def ev_block_stop(index: int) -> SimpleNamespace:
    return SimpleNamespace(type="content_block_stop", index=index)


def ev_text(index: int, text: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="content_block_delta",
        index=index,
        delta=SimpleNamespace(type="text_delta", text=text),
    )


def ev_thinking(index: int) -> SimpleNamespace:
    return SimpleNamespace(
        type="content_block_delta",
        index=index,
        delta=SimpleNamespace(type="thinking_delta", thinking=""),
    )


def ev_cite(index: int, search_result_index: int, cited_text: str) -> SimpleNamespace:
    citation = SimpleNamespace(
        type="search_result_location",
        search_result_index=search_result_index,
        cited_text=cited_text,
    )
    return SimpleNamespace(
        type="content_block_delta",
        index=index,
        delta=SimpleNamespace(type="citations_delta", citation=citation),
    )


def final_message(
    stop_reason: str = "end_turn",
    model: str = "claude-opus-5-5",
    input_tokens: int = 1000,
    output_tokens: int = 100,
) -> SimpleNamespace:
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    return SimpleNamespace(stop_reason=stop_reason, model=model, usage=usage)


class _FakeStream:
    def __init__(self, events, message, error):
        self._events, self._message, self._error = events, message, error

    async def __aenter__(self):
        if self._error is not None:
            raise self._error
        return self

    async def __aexit__(self, *exc):
        return False

    async def __aiter__(self):
        for event in self._events:
            yield event

    async def get_final_message(self):
        return self._message


class FakeClaude:
    """`client.beta.messages.stream(**kwargs)` replaying scripted events."""

    def __init__(self, events=(), message=None, error: Exception | None = None) -> None:
        self.events = list(events)
        self.message = message or final_message()
        self.error = error
        self.requests: list[dict] = []
        self.closed = False
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **kwargs):
        self.requests.append(kwargs)
        return _FakeStream(self.events, self.message, self.error)

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_claude_cls() -> type[FakeClaude]:
    return FakeClaude
