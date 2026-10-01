import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class Embedder(Protocol):
    dim: int

    def embed_passages(self, passages: Sequence[tuple[str, str]]) -> list[list[float]]:
        """Embed (document title, chunk text) pairs."""
        ...

    def embed_query(self, query: str) -> list[float]: ...


@dataclass(frozen=True)
class ModelSpec:
    """A fastembed model and the input format it was trained with.

    fastembed doesn't add task prefixes itself, so they live here.
    """

    name: str
    dim: int
    query_template: str = "{query}"
    passage_template: str = "{text}"


MODELS: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in [
        ModelSpec("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 384),
        ModelSpec("minishlab/potion-multilingual-128M", 256),
        ModelSpec(
            "google/embeddinggemma-300m",
            768,
            query_template="task: search result | query: {query}",
            passage_template="title: {title} | text: {text}",
        ),
        ModelSpec(
            "Qwen/Qwen3-Embedding-0.6B-Q",
            1024,
            query_template=(
                "Instruct: Given a question, retrieve passages from company documents "
                "that answer it\nQuery:{query}"
            ),
        ),
    ]
}


class FastEmbedder:
    """Local ONNX embeddings via fastembed. The model loads on first use."""

    def __init__(
        self, model_name: str, cache_dir: Path | None = None, model_path: Path | None = None
    ) -> None:
        """`model_path`: load from this directory instead of downloading into `cache_dir`."""
        if model_name not in MODELS:
            raise ValueError(f"Unsupported embedding model: {model_name}")
        self.spec = MODELS[model_name]
        self.dim = self.spec.dim
        self._cache_dir = cache_dir
        self._model_path = model_path
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding  # heavy import, only when needed

                self._model = TextEmbedding(
                    self.spec.name,
                    cache_dir=str(self._cache_dir) if self._cache_dir else None,
                    specific_model_path=str(self._model_path) if self._model_path else None,
                )
            return self._model

    def embed_passages(self, passages: Sequence[tuple[str, str]]) -> list[list[float]]:
        texts = [
            self.spec.passage_template.format(title=title or "none", text=text)
            for title, text in passages
        ]
        return [vector.tolist() for vector in self._get_model().embed(texts)]

    def embed_query(self, query: str) -> list[float]:
        text = self.spec.query_template.format(query=query)
        return next(iter(self._get_model().embed([text]))).tolist()
