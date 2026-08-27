"""Shared fixtures. Integration tests run against the project's real dev
Postgres/Qdrant/Ollama (no mocks, no separate test containers - see
ARCHITECTURE.md section 16/17: this project's own manual verification
throughout Phases 1-2 was always done against these same real services,
and the user asked for tests to follow that pattern rather than mock it
away). Everything a test creates is prefixed `pytest_` and torn down
immediately after, whether the test passed or failed.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.chunking import GenericChunker, StructureAwareChunker
from app.config import get_settings
from app.embeddings import OllamaEmbeddingClient
from app.llm import OllamaGenerationClient
from app.reranking import LazyReranker
from app.retrieval import QdrantStore, SparseEmbedder
from app.services import ClientWorkspaceService

TEST_PREFIX = "pytest_"


def test_name(suffix: str = "") -> str:
    return f"{TEST_PREFIX}{uuid.uuid4().hex[:10]}{suffix}"


@pytest.fixture
def unique_name() -> str:
    return test_name()


@pytest.fixture(scope="session")
def settings():
    return get_settings()


@pytest_asyncio.fixture
async def db_session() -> AsyncIterator[AsyncSession]:
    # A fresh engine per test function, not app.db.base's process-lifetime
    # singleton: that engine's asyncpg connection pool binds connections to
    # whichever event loop created them, and pytest-asyncio gives every
    # test function its own loop by default - reusing the singleton across
    # tests hands a later test a connection bound to an already-closed
    # loop from an earlier one ("attached to a different loop").
    engine = create_async_engine(get_settings().postgres_dsn)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        yield session
    await engine.dispose()


# Everything below that wraps an async HTTP client (Ollama, Qdrant) is
# deliberately function-scoped, not session-scoped: pytest-asyncio gives
# every test function its own event loop by default, and a long-lived
# async client's underlying connection is bound to whichever loop was
# running when it was first used - sharing one across tests hits the same
# "attached to a different loop" class of failure `db_session` avoids
# above. `sparse_embedder`/`generic_chunker`/`settings` hold no event-loop
# state (sync-only), so those stay session-scoped to skip re-loading the
# ONNX model per test.
@pytest.fixture
def dense_client(settings) -> OllamaEmbeddingClient:
    return OllamaEmbeddingClient(base_url=settings.ollama_base_url, model=settings.embedding_model)


@pytest.fixture(scope="session")
def sparse_embedder() -> SparseEmbedder:
    return SparseEmbedder()


@pytest.fixture(scope="session")
def generic_chunker(settings) -> GenericChunker:
    return GenericChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)


@pytest.fixture(scope="session")
def standard_chunker(settings) -> StructureAwareChunker:
    return StructureAwareChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)


@pytest.fixture
def standard_store(settings) -> QdrantStore:
    return QdrantStore(url=settings.qdrant_url, collection_name=settings.qdrant_collection)


@pytest_asyncio.fixture
async def reranker(settings) -> AsyncIterator[LazyReranker]:
    instance = LazyReranker(
        model_name=settings.reranker_model,
        device=settings.reranker_device,
        idle_timeout_seconds=settings.reranker_idle_timeout_seconds,
    )
    yield instance
    # If a test triggered a model load, `_ensure_loaded` leaves its
    # idle-unload watcher task running on this test's event loop forever -
    # fine in production (the app's loop never closes), but here the loop
    # closes at test end, and an unfinished task on a closing loop prints
    # "Task was destroyed but it is pending!" - cancel it explicitly.
    if instance._idle_task is not None:
        instance._idle_task.cancel()


@pytest.fixture
def generation_client(settings) -> OllamaGenerationClient:
    return OllamaGenerationClient(
        base_url=settings.ollama_base_url, model=settings.ollama_model, num_ctx=settings.ollama_num_ctx
    )


@pytest_asyncio.fixture
async def qdrant_client(settings) -> AsyncIterator[AsyncQdrantClient]:
    client = AsyncQdrantClient(url=settings.qdrant_url)
    yield client
    await client.close()


async def _sweep_stale_pytest_data() -> None:
    """Best-effort cleanup of anything left behind by a test process that
    got killed mid-run (a timeout, Ctrl-C) before its own fixture teardown
    could run - `client_and_project`'s finally block handles the common
    case, this is only the safety net for when that didn't get to run.
    Plain asyncio, not a pytest-asyncio fixture: it runs from the sync
    `pytest_sessionstart`/`pytest_sessionfinish` hooks below (once per
    whole run, before pytest-asyncio's own per-test loop machinery is
    involved at all) via its own throwaway `asyncio.run` loop."""
    settings = get_settings()
    engine = create_async_engine(settings.postgres_dsn)
    try:
        async with engine.connect() as conn:
            result = await conn.execute(
                text("SELECT qdrant_collection_name FROM clients WHERE name LIKE :prefix"),
                {"prefix": f"{TEST_PREFIX}%"},
            )
            stale_collections = [row[0] for row in result.all()]
            if not stale_collections:
                return
            await conn.execute(text("DELETE FROM clients WHERE name LIKE :prefix"), {"prefix": f"{TEST_PREFIX}%"})
            await conn.commit()
    finally:
        await engine.dispose()

    qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    try:
        for collection_name in stale_collections:
            if await qdrant.collection_exists(collection_name):
                await qdrant.delete_collection(collection_name)
    finally:
        await qdrant.close()


def pytest_sessionstart(session: pytest.Session) -> None:
    asyncio.run(_sweep_stale_pytest_data())


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    asyncio.run(_sweep_stale_pytest_data())


@pytest_asyncio.fixture
async def client_and_project(db_session: AsyncSession) -> AsyncIterator[tuple]:
    """A throwaway Client + AuditProject, deleted (cascades to
    projects/findings/requirement statuses/evidence) after the test; the
    client's Qdrant collection is deleted too, since `Client.delete()`
    doesn't know about Qdrant."""
    service = ClientWorkspaceService(db_session)
    client = await service.create_client(test_name())
    project = await service.create_project(client.id, "Test Audit")
    try:
        yield client, project
    finally:
        await db_session.execute(text("DELETE FROM clients WHERE id = :id"), {"id": str(client.id)})
        await db_session.commit()
        qdrant = AsyncQdrantClient(url=get_settings().qdrant_url)
        try:
            if await qdrant.collection_exists(client.qdrant_collection_name):
                await qdrant.delete_collection(client.qdrant_collection_name)
        finally:
            await qdrant.close()
