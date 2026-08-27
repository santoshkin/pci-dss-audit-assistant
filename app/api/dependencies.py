"""Process-wide singletons for FastAPI dependency injection. The Ollama
embedding client and FastEmbed sparse model are expensive enough to set up
(network client init, ONNX model load) that constructing them per request
would add real latency to every evidence upload - cached once per process
instead, mirroring `app.config.get_settings`'s `lru_cache`.
"""

from __future__ import annotations

from functools import lru_cache

from app.chunking import GenericChunker, StructureAwareChunker
from app.config import get_settings
from app.embeddings import OllamaEmbeddingClient
from app.llm import OllamaGenerationClient
from app.reranking import LazyReranker
from app.retrieval import QdrantStore, SparseEmbedder


@lru_cache
def get_dense_client() -> OllamaEmbeddingClient:
    settings = get_settings()
    return OllamaEmbeddingClient(base_url=settings.ollama_base_url, model=settings.embedding_model)


@lru_cache
def get_sparse_embedder() -> SparseEmbedder:
    return SparseEmbedder()


@lru_cache
def get_generic_chunker() -> GenericChunker:
    settings = get_settings()
    return GenericChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)


@lru_cache
def get_standard_chunker() -> StructureAwareChunker:
    settings = get_settings()
    return StructureAwareChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)


@lru_cache
def get_standard_store() -> QdrantStore:
    settings = get_settings()
    return QdrantStore(url=settings.qdrant_url, collection_name=settings.qdrant_collection)


@lru_cache
def get_reranker() -> LazyReranker:
    settings = get_settings()
    return LazyReranker(
        model_name=settings.reranker_model,
        device=settings.reranker_device,
        idle_timeout_seconds=settings.reranker_idle_timeout_seconds,
    )


@lru_cache
def get_generation_client() -> OllamaGenerationClient:
    settings = get_settings()
    return OllamaGenerationClient(
        base_url=settings.ollama_base_url, model=settings.ollama_model, num_ctx=settings.ollama_num_ctx
    )
