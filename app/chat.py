"""Phase 0 MVP entry point (PLAN.md sections 26-27):

    python -m app.chat "Какие требования PCI DSS относятся к MFA?"

Retrieval pipeline: exact Requirement ID lookup (PLAN.md section 9) +
hybrid dense/sparse search (app/retrieval), reranked (app/reranking), fed
to Ollama generation with the PCI DSS Assistant system prompt
(app/prompts). Citations are appended from the retrieved chunks' own
metadata, never left to the model to invent (PLAN.md section 12).

`format_source`/`format_context_entry`/`strip_model_sources_section` are
public (not module-private) because `app/services/project_chat.py` (Phase
2's client-context chat) reuses them as-is against a merged shared+client
result set, rather than duplicating this formatting logic.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from app.config import get_settings
from app.embeddings import OllamaEmbeddingClient
from app.llm import OllamaGenerationClient
from app.prompts import SYSTEM_PROMPT
from app.reranking import LazyReranker
from app.retrieval import QdrantStore, SparseEmbedder
from app.retrieval.requirement_lookup import extract_requirement_id_candidates

NO_INFO_MESSAGE = "Недостаточно информации в доступных источниках."


def format_source(payload: dict[str, Any]) -> str:
    version = payload.get("version")
    version_known = version and version != "unknown"

    if payload.get("requirement_id"):
        # PCI DSS Standard chunk (app/chunking.StructureAwareChunker).
        parts = [f"PCI DSS v{version}" if version_known else "PCI DSS", f"Requirement {payload['requirement_id']}"]
        if payload.get("testing_procedure_ids"):
            parts.append(f"Testing Procedure {', '.join(payload['testing_procedure_ids'])}")
        if payload.get("subsection"):
            parts.append(str(payload["subsection"]))
    elif payload.get("document_type") == "evidence":
        # Client evidence chunk (app/services/evidence_intake.py) - never
        # confuse this with an official Standard/FAQ source in citations.
        parts = [f"Evidence: {payload['document_title']}"]
        if payload.get("section"):
            parts.append(str(payload["section"]))
    else:
        # Generic chunk (app/chunking.GenericChunker) - FAQ/Guidance PDF or
        # an org .docx: no requirement/TP, cite by document + section.
        parts = [payload["document_title"]]
        if version_known:
            parts.append(f"v{version}")
        if payload.get("section"):
            parts.append(str(payload["section"]))

    page_start, page_end = payload.get("page_start", 0), payload.get("page_end", 0)
    if page_start:
        parts.append(f"p. {page_start}" if page_start == page_end else f"pp. {page_start}-{page_end}")
    return ", ".join(parts)


def format_context_entry(payload: dict[str, Any]) -> str:
    return f"[{format_source(payload)}]\n{payload['text']}"


def strip_model_sources_section(generated: str) -> str:
    """The system prompt tells the model not to write its own "##
    Источники" section (citations must come from chunk metadata, never the
    model - PLAN.md section 12), but models don't always comply. Drop
    anything from that heading onward rather than trust it."""
    marker = generated.lower().rfind("## источники")
    return generated[:marker].rstrip() if marker != -1 else generated


async def answer(question: str) -> str:
    settings = get_settings()
    dense_client = OllamaEmbeddingClient(base_url=settings.ollama_base_url, model=settings.embedding_model)
    sparse_embedder = SparseEmbedder()
    store = QdrantStore(url=settings.qdrant_url, collection_name=settings.qdrant_collection)
    reranker = LazyReranker(
        model_name=settings.reranker_model,
        device=settings.reranker_device,
        idle_timeout_seconds=settings.reranker_idle_timeout_seconds,
    )
    gen_client = OllamaGenerationClient(
        base_url=settings.ollama_base_url, model=settings.ollama_model, num_ctx=settings.ollama_num_ctx
    )

    seen_point_ids: set[Any] = set()
    payloads: list[dict[str, Any]] = []

    for requirement_id in extract_requirement_id_candidates(question):
        for record in await store.get_by_requirement_id(requirement_id):
            if record.id not in seen_point_ids:
                seen_point_ids.add(record.id)
                payloads.append(record.payload)

    query_dense = await dense_client.embed_one(question)
    query_sparse = sparse_embedder.embed_one(question)
    results = await store.search(query_dense, query_sparse, top_k=settings.top_k)

    if results:
        scores = await reranker.rerank(question, [r.payload["text"] for r in results])
        ranked = sorted(zip(results, scores), key=lambda pair: pair[1], reverse=True)
        for record, _score in ranked[: settings.rerank_top_k]:
            if record.id not in seen_point_ids:
                seen_point_ids.add(record.id)
                payloads.append(record.payload)

    if not payloads:
        return NO_INFO_MESSAGE

    context = "\n\n---\n\n".join(format_context_entry(p) for p in payloads)
    user_prompt = f"Контекст:\n\n{context}\n\nВопрос пользователя: {question}"
    generated = strip_model_sources_section(await gen_client.generate(SYSTEM_PROMPT, user_prompt))

    sources_block = "\n".join(f"* {s}" for s in sorted({format_source(p) for p in payloads}))
    return f"{generated}\n\n## Источники\n\n{sources_block}"


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m app.chat "question"')
        raise SystemExit(1)
    print(asyncio.run(answer(sys.argv[1])))


if __name__ == "__main__":
    main()
