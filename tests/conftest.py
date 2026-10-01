import hashlib
from collections.abc import Sequence

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
