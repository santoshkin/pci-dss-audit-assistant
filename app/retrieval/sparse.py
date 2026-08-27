"""Sparse (keyword-style) embeddings for Qdrant's native hybrid search.

Ollama has no BM25/sparse embedding endpoint, so sparse vectors are
computed client-side via FastEmbed's `Qdrant/bm25` model - a local ONNX
model, no network calls once its (small) weights are cached, keeping the
whole pipeline local per the original spec. See ARCHITECTURE.md section 1a
for why native Qdrant sparse+dense fusion was chosen over a separate BM25
engine.
"""

from __future__ import annotations

from fastembed import SparseTextEmbedding

SparseVector = tuple[list[int], list[float]]


class SparseEmbedder:
    def __init__(self, model_name: str = "Qdrant/bm25") -> None:
        self._model = SparseTextEmbedding(model_name=model_name)

    def embed(self, texts: list[str]) -> list[SparseVector]:
        return [(e.indices.tolist(), e.values.tolist()) for e in self._model.embed(texts)]

    def embed_one(self, text: str) -> SparseVector:
        return self.embed([text])[0]
