"""Embedding client backed by Ollama's `/api/embed` endpoint.

The embedding model is read from config (`EMBEDDING_MODEL` in `.env`), never
hardcoded, per the original spec's requirement to keep model choice
configurable (PLAN.md section 4). Confirmed working against `bge-m3` on the
dev machine's Ollama instance (ARCHITECTURE.md section 1) - 1024-dim
vectors, 8192-token context, well above any real chunk length produced by
`app/chunking` (max observed ~2.3k characters).
"""

from __future__ import annotations

import ollama


class OllamaEmbeddingClient:
    def __init__(self, base_url: str, model: str, batch_size: int = 32) -> None:
        self._client = ollama.AsyncClient(host=base_url)
        self.model = model
        self.batch_size = batch_size

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embeds `texts` in request-size batches, preserving input order."""
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = await self._client.embed(model=self.model, input=batch)
            if len(response.embeddings) != len(batch):
                raise RuntimeError(
                    f"Ollama returned {len(response.embeddings)} embeddings for a batch of "
                    f"{len(batch)} inputs (model={self.model!r})"
                )
            embeddings.extend([list(e) for e in response.embeddings])
        return embeddings

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]
