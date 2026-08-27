"""Client-context chat (PLAN.md section 3, Phase 1 note "работа в
контексте клиента" — deferred in `app/api/app.py` until Evidence intake
(Phase 2) gave each client an actual Qdrant collection to search). Same
retrieval/generation pipeline as `app/chat.py`'s Phase 0 CLI, but merges
hits from the project's client evidence collection alongside the shared
Standard collection before reranking, so an audit question can draw on
both what the Standard says and what evidence has actually been collected
for this client - `app/chat.py`'s citation formatting already tags
evidence chunks distinctly (`"Evidence: ..."` vs `"PCI DSS Requirement
..."`), so the two never get confused in the answer's sources section.

**Retrieval expansion (ARCHITECTURE.md section 21):** plain
question-embedding similarity was found to reliably miss the right
context for broad "does X comply with PCI DSS" questions - a single
embedding pass rarely surfaces every sibling sub-requirement (e.g. finds
8.3.6 but not 8.3.4/8.3.7/8.3.9), and the client's own policy clause text
("минимум 8 символов") is lexically much closer to the *Standard's* own
clause text ("Minimum length of 12 characters") than to the user's
abstract question, so evidence retrieval driven only by the question
misses it too. Two deterministic expansion steps fix this without any
topic-specific hardcoding:

1. Any requirement_id surfaced by exact lookup or hybrid search pulls in
   its whole 2-level sibling family (`requirement_family_prefix`) via
   exact metadata lookup, not embedding luck.
2. Each of those Standard chunks' own text is used as an *additional*
   search query against the client's evidence collection - so evidence
   retrieval isn't limited to what matches the user's phrasing.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.chat import NO_INFO_MESSAGE, format_context_entry, format_source, strip_model_sources_section
from app.db.models import AuditProject, Client
from app.embeddings import OllamaEmbeddingClient
from app.llm import OllamaGenerationClient
from app.prompts import SYSTEM_PROMPT
from app.reranking import LazyReranker
from app.retrieval import QdrantStore, SparseEmbedder
from app.retrieval.requirement_lookup import extract_requirement_id_candidates, requirement_family_prefix

# Cap on evidence-side expansion searches (one per expanded Standard
# chunk) - PCI DSS's own 2-level sections are small enough in practice
# (~5-15 sub-requirements) that this rarely binds, it just guards against
# a pathological family blowing up the number of Ollama embed calls.
MAX_EXPANSION_QUERIES = 15
EXPANSION_SEARCH_TOP_K = 5


class ProjectChatService:
    def __init__(
        self,
        session: AsyncSession,
        dense_client: OllamaEmbeddingClient,
        sparse_embedder: SparseEmbedder,
        reranker: LazyReranker,
        gen_client: OllamaGenerationClient,
        standard_store: QdrantStore,
        qdrant_url: str,
        top_k: int,
        rerank_top_k: int,
    ) -> None:
        self.session = session
        self.dense_client = dense_client
        self.sparse_embedder = sparse_embedder
        self.reranker = reranker
        self.gen_client = gen_client
        self.standard_store = standard_store
        self.qdrant_url = qdrant_url
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k

    async def _expand_requirement_families(self, found_ids: set[str]) -> list[Any]:
        """Given requirement IDs already surfaced (by exact lookup or
        hybrid search), fetches every sibling under the same 2-level
        family that wasn't already found, in parallel."""
        family_prefixes = {requirement_family_prefix(rid) for rid in found_ids}
        if not family_prefixes:
            return []
        all_ids = await self.standard_store.list_requirement_ids()
        sibling_ids = [
            rid for rid in all_ids if rid not in found_ids and requirement_family_prefix(rid) in family_prefixes
        ]
        if not sibling_ids:
            return []
        results = await asyncio.gather(*(self.standard_store.get_by_requirement_id(rid) for rid in sibling_ids))
        return [record for records in results for record in records]

    async def _expand_evidence_via_standard_text(self, client_store: QdrantStore, standard_texts: list[str]) -> list[Any]:
        """Searches the client's evidence collection using each given
        Standard chunk's own text as the query, instead of the user's
        question - see module docstring for why this is necessary."""
        texts = standard_texts[:MAX_EXPANSION_QUERIES]
        if not texts:
            return []
        dense_vectors = await self.dense_client.embed(texts)
        sparse_vectors = self.sparse_embedder.embed(texts)
        results = await asyncio.gather(
            *(
                client_store.search(dense_vec, sparse_vec, top_k=EXPANSION_SEARCH_TOP_K)
                for dense_vec, sparse_vec in zip(dense_vectors, sparse_vectors)
            )
        )
        return [record for records in results for record in records]

    async def answer(self, project_id: uuid.UUID, question: str) -> str | None:
        """Returns None if `project_id` doesn't exist (caller maps that to
        404); otherwise the generated answer, or `NO_INFO_MESSAGE`."""
        project = await self.session.get(AuditProject, project_id)
        if project is None:
            return None
        client = await self.session.get(Client, project.client_id)
        assert client is not None

        client_store = QdrantStore(url=self.qdrant_url, collection_name=client.qdrant_collection_name)
        # The client's collection may not exist yet if no evidence has been
        # uploaded for them - ensure_collection is a cheap no-op otherwise.
        await client_store.ensure_collection()

        seen_point_ids: set[Any] = set()
        payloads: list[dict[str, Any]] = []

        def take(records: list[Any]) -> None:
            for record in records:
                if record.id not in seen_point_ids:
                    seen_point_ids.add(record.id)
                    payloads.append(record.payload)

        exact_hits: list[Any] = []
        for requirement_id in extract_requirement_id_candidates(question):
            exact_hits.extend(await self.standard_store.get_by_requirement_id(requirement_id))
        take(exact_hits)

        query_dense = await self.dense_client.embed_one(question)
        query_sparse = self.sparse_embedder.embed_one(question)

        standard_results = await self.standard_store.search(query_dense, query_sparse, top_k=self.top_k)
        evidence_results = await client_store.search(query_dense, query_sparse, top_k=self.top_k)

        found_requirement_ids = {r.payload.get("requirement_id") for r in list(exact_hits) + list(standard_results)}
        found_requirement_ids.discard(None)
        family_chunks = await self._expand_requirement_families(found_requirement_ids)
        expansion_evidence = await self._expand_evidence_via_standard_text(
            client_store, [r.payload["text"] for r in family_chunks]
        )

        combined = list(standard_results) + list(evidence_results) + family_chunks + expansion_evidence
        deduped: list[Any] = []
        combined_seen = set(seen_point_ids)
        for record in combined:
            if record.id in combined_seen:
                continue
            combined_seen.add(record.id)
            deduped.append(record)

        if deduped:
            scores = await self.reranker.rerank(question, [r.payload["text"] for r in deduped])
            ranked = sorted(zip(deduped, scores), key=lambda pair: pair[1], reverse=True)
            take([record for record, _score in ranked[: self.rerank_top_k]])

        if not payloads:
            return NO_INFO_MESSAGE

        context = "\n\n---\n\n".join(format_context_entry(p) for p in payloads)
        user_prompt = f"Контекст:\n\n{context}\n\nВопрос пользователя: {question}"
        generated = strip_model_sources_section(await self.gen_client.generate(SYSTEM_PROMPT, user_prompt))

        sources_block = "\n".join(f"* {s}" for s in sorted({format_source(p) for p in payloads}))
        return f"{generated}\n\n## Источники\n\n{sources_block}"
