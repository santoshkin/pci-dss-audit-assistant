"""End-to-end smoke test: ingestion -> chunking -> embeddings, on the real
PCI DSS v4.0.1 document. Confirms embeddings work over every real chunk
(not just synthetic text) and does a basic semantic sanity check (nearest
chunks to a query should actually be on-topic) before any Qdrant/retrieval
code is built on top of this.

Usage: .venv/bin/python scripts/embedding_validation.py
"""

import asyncio
import time
from pathlib import Path

from app.chunking import StructureAwareChunker
from app.config import get_settings
from app.embeddings import OllamaEmbeddingClient
from app.ingestion import PdfLayoutParser, StructureExtractor

PDF_PATH = Path("data/documents/pci_dss/PCI-DSS-v4_0_1.pdf")


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb)


async def main() -> None:
    settings = get_settings()
    parser = PdfLayoutParser(PDF_PATH)
    metadata = parser.extract_metadata()
    pages = list(parser.iter_pages())
    section_titles = parser.extract_section_titles()
    extraction = StructureExtractor().extract(pages, section_titles)

    chunker = StructureAwareChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    chunks = chunker.chunk(
        document_id=f"pci_dss_v{metadata.version}",
        document_title=metadata.title,
        document_type="pci_dss",
        version=metadata.version,
        publication_date=metadata.date,
        source_file=str(PDF_PATH),
        requirements=extraction.requirements,
        testing_procedures=extraction.testing_procedures,
        guidance_blocks=extraction.guidance_blocks,
    )
    print(f"Chunks to embed: {len(chunks)}")

    client = OllamaEmbeddingClient(base_url=settings.ollama_base_url, model=settings.embedding_model)
    start = time.time()
    embeddings = await client.embed([c.text for c in chunks])
    elapsed = time.time() - start
    print(f"Embedded {len(embeddings)} chunks in {elapsed:.1f}s ({elapsed / len(chunks) * 1000:.1f}ms/chunk)")
    assert len(embeddings) == len(chunks)
    assert all(len(e) == 1024 for e in embeddings)

    queries = [
        "multi-factor authentication requirements",
        "penetration testing methodology",
        "storing sensitive authentication data",
    ]
    for query in queries:
        query_emb = await client.embed_one(query)
        scored = sorted(
            zip(chunks, embeddings), key=lambda pair: cosine(query_emb, pair[1]), reverse=True
        )
        print()
        print(f"=== Query: {query!r} — top 5 ===")
        for chunk, _ in scored[:5]:
            print(f"[{chunk.requirement_id}/{chunk.chunk_type}/{chunk.subsection}] {chunk.text[:100]}")


if __name__ == "__main__":
    asyncio.run(main())
