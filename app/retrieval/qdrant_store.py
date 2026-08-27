"""Qdrant-backed hybrid (dense + sparse) store, see ARCHITECTURE.md
section 1a: native Query API fusion in one collection instead of a
separate BM25 index, so incremental re-indexing never has to keep two
stores in sync (original spec item 18).
"""

from __future__ import annotations

import uuid
from dataclasses import asdict

from qdrant_client import AsyncQdrantClient, models

from app.chunking import Chunk
from app.retrieval.sparse import SparseVector

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

# Fixed namespace so the same chunk.id always maps to the same Qdrant point
# id across re-ingestion runs (upsert overwrites rather than duplicates).
_POINT_ID_NAMESPACE = uuid.UUID("6f6f1e0a-6f1b-4b8e-9c3a-9a9f2f6a2b31")


def chunk_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, chunk_id))


class QdrantStore:
    def __init__(self, url: str, collection_name: str, dense_vector_size: int = 1024, timeout: float = 60.0) -> None:
        # 30s used to be plenty, but `create_collection` alone was
        # observed taking ~15s under real load on the shared dev Qdrant
        # host (see ARCHITECTURE.md section 22) - 60s gives headroom
        # instead of a hard-to-diagnose ReadTimeout under normal jitter.
        self.client = AsyncQdrantClient(url=url, timeout=timeout)
        self.collection_name = collection_name
        self.dense_vector_size = dense_vector_size
        self._requirement_ids_cache: list[str] | None = None

    async def ensure_collection(self) -> None:
        if await self.client.collection_exists(self.collection_name):
            return
        await self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(size=self.dense_vector_size, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={SPARSE_VECTOR_NAME: models.SparseVectorParams()},
        )

    async def upsert(
        self,
        chunks: list[Chunk],
        dense_embeddings: list[list[float]],
        sparse_embeddings: list[SparseVector],
    ) -> None:
        points = [
            models.PointStruct(
                id=chunk_point_id(chunk.id),
                vector={
                    DENSE_VECTOR_NAME: dense,
                    SPARSE_VECTOR_NAME: models.SparseVector(indices=indices, values=values),
                },
                payload=asdict(chunk),
            )
            for chunk, dense, (indices, values) in zip(chunks, dense_embeddings, sparse_embeddings)
        ]
        await self.client.upsert(collection_name=self.collection_name, points=points)

    async def delete_by_document_id(self, document_id: str) -> None:
        """Deletes every chunk belonging to one ingested document (matched
        by its `document_id` payload field) - used when an Evidence record
        is deleted via the API, so its chunks don't linger as orphaned
        points in the client's collection."""
        if not await self.client.collection_exists(self.collection_name):
            return
        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
                )
            ),
        )

    def invalidate_requirement_ids_cache(self) -> None:
        """Call after upserting new chunks (e.g. a knowledge-base document
        upload via API) so `list_requirement_ids()`'s next call re-scrolls
        instead of serving a stale catalog missing the new IDs."""
        self._requirement_ids_cache = None

    async def get_by_requirement_id(self, requirement_id: str) -> list[models.Record]:
        """Exact metadata lookup (PLAN.md section 9), bypassing vector
        search entirely - used when the query names a specific requirement."""
        records, _next_offset = await self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="requirement_id", match=models.MatchValue(value=requirement_id))]
            ),
            limit=100,
        )
        return records

    async def list_requirement_ids(self) -> list[str]:
        """All distinct non-null `requirement_id` values in this
        collection - used by `app/services/project_chat.py` to expand a
        found requirement to its whole sibling family. Cached for the
        life of this instance: the Standard corpus this runs against
        changes rarely (re-ingested only on a new PCI DSS release, see
        ARCHITECTURE.md's accuracy/speed tradeoff), so a full-collection
        scroll on every question would be wasteful."""
        if self._requirement_ids_cache is not None:
            return self._requirement_ids_cache

        seen: set[str] = set()
        next_offset = None
        while True:
            records, next_offset = await self.client.scroll(
                collection_name=self.collection_name,
                with_payload=["requirement_id"],
                with_vectors=False,
                limit=500,
                offset=next_offset,
            )
            for record in records:
                requirement_id = record.payload.get("requirement_id")
                if requirement_id:
                    seen.add(requirement_id)
            if next_offset is None:
                break

        self._requirement_ids_cache = sorted(seen)
        return self._requirement_ids_cache

    async def search(
        self,
        query_dense: list[float],
        query_sparse: SparseVector,
        top_k: int,
        prefetch_limit: int = 50,
    ) -> list[models.ScoredPoint]:
        indices, values = query_sparse
        result = await self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(query=query_dense, using=DENSE_VECTOR_NAME, limit=prefetch_limit),
                models.Prefetch(
                    query=models.SparseVector(indices=indices, values=values),
                    using=SPARSE_VECTOR_NAME,
                    limit=prefetch_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
        )
        return result.points
