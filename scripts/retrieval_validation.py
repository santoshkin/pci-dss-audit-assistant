"""End-to-end smoke test: ingestion -> chunking -> embeddings -> Qdrant
hybrid (dense+sparse, RRF) upsert and search, on the real PCI DSS v4.0.1
document. Confirms the full pipeline before any FastAPI/CLI layer is built
on top of it.

Usage: .venv/bin/python scripts/retrieval_validation.py
"""

import asyncio
import time
from pathlib import Path

from app.chunking import StructureAwareChunker
from app.config import get_settings
from app.embeddings import OllamaEmbeddingClient
from app.ingestion import PdfLayoutParser, StructureExtractor
from app.retrieval import QdrantStore, SparseEmbedder

PDF_PATH = Path("data/documents/pci_dss/PCI-DSS-v4_0_1.pdf")


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
    print(f"Chunks: {len(chunks)}")

    dense_client = OllamaEmbeddingClient(base_url=settings.ollama_base_url, model=settings.embedding_model)
    sparse_embedder = SparseEmbedder()

    start = time.time()
    dense_embeddings = await dense_client.embed([c.text for c in chunks])
    sparse_embeddings = sparse_embedder.embed([c.text for c in chunks])
    print(f"Embedded (dense+sparse) in {time.time() - start:.1f}s")

    store = QdrantStore(url=settings.qdrant_url, collection_name=settings.qdrant_collection)
    await store.ensure_collection()
    await store.upsert(chunks, dense_embeddings, sparse_embeddings)
    print(f"Upserted {len(chunks)} points into {settings.qdrant_collection!r} @ {settings.qdrant_url}")

    queries = [
        "multi-factor authentication requirements",
        "8.3.5",  # exact requirement ID - sparse should nail this
        "how often should penetration testing be performed",
    ]
    for query in queries:
        query_dense = await dense_client.embed_one(query)
        query_sparse = sparse_embedder.embed_one(query)
        results = await store.search(query_dense, query_sparse, top_k=5)
        print()
        print(f"=== Query: {query!r} — top 5 (hybrid RRF) ===")
        for r in results:
            p = r.payload
            print(f"score={r.score:.4f} [{p['requirement_id']}/{p['chunk_type']}/{p['subsection']}] {p['text'][:100]}")


if __name__ == "__main__":
    asyncio.run(main())
