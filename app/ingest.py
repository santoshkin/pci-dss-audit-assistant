"""Phase 0 MVP entry point (PLAN.md sections 26-27):

    python -m app.ingest

Indexes every PDF under `data/documents/pci_dss/` with the coordinate-based
Standard parser (`app/ingestion`/`app/chunking.StructureAwareChunker`), and
everything under `data/documents/faq/`, `data/documents/guidance/` (PDF)
with the lighter-weight generic path (`app/chunking.GenericChunker`) - see
ARCHITECTURE.md section 14 for why these use different parsers (arbitrary
format/high volume vs. the Standard's rarely-updated fixed template).

`data/documents/org/` is NOT indexed here - it holds example CLIENT
evidence documents, which belong in a per-client Qdrant collection
(PLAN.md section 2, "Мультиклиентность и изоляция"), never this shared
one. Evidence intake is PLAN.md's Phase 2, not yet built; when it is,
`ingest_docx`/`ingest_generic_pdf` below are reusable as-is against a
per-client `QdrantStore` instead of the shared one.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from app.chunking import Chunk, GenericChunker, StructureAwareChunker
from app.config import get_settings
from app.embeddings import OllamaEmbeddingClient
from app.ingestion import DocxParser, GenericPdfParser, PdfLayoutParser, StructureExtractor
from app.retrieval import QdrantStore, SparseEmbedder


async def ingest_pdf(
    path: Path,
    dense_client: OllamaEmbeddingClient,
    sparse_embedder: SparseEmbedder,
    store: QdrantStore,
    chunker: StructureAwareChunker,
) -> list[Chunk]:
    parser = PdfLayoutParser(path)
    metadata = parser.extract_metadata()
    pages = list(parser.iter_pages())
    section_titles = parser.extract_section_titles()
    extraction = StructureExtractor().extract(pages, section_titles)

    for warning in extraction.warnings:
        print(f"  warning: {warning}")

    chunks = chunker.chunk(
        document_id=f"pci_dss_v{metadata.version}",
        document_title=metadata.title,
        document_type="pci_dss",
        version=metadata.version,
        publication_date=metadata.date,
        source_file=str(path),
        requirements=extraction.requirements,
        testing_procedures=extraction.testing_procedures,
        guidance_blocks=extraction.guidance_blocks,
    )

    dense_embeddings = await dense_client.embed([c.text for c in chunks])
    sparse_embeddings = sparse_embedder.embed([c.text for c in chunks])
    await store.upsert(chunks, dense_embeddings, sparse_embeddings)
    return chunks


async def ingest_generic_pdf(
    path: Path,
    document_type: str,
    dense_client: OllamaEmbeddingClient,
    sparse_embedder: SparseEmbedder,
    store: QdrantStore,
    chunker: GenericChunker,
) -> list[Chunk]:
    parser = GenericPdfParser(path)
    pages = parser.extract_pages()

    chunks = chunker.chunk_pdf(
        pages=pages,
        document_id=f"{document_type}_{path.stem}",
        document_title=parser.guess_title(pages),
        document_type=document_type,
        version=parser.guess_version(pages),
        publication_date=parser.guess_date(pages),
        source_file=str(path),
    )

    dense_embeddings = await dense_client.embed([c.text for c in chunks])
    sparse_embeddings = sparse_embedder.embed([c.text for c in chunks])
    await store.upsert(chunks, dense_embeddings, sparse_embeddings)
    return chunks


async def ingest_docx(
    path: Path,
    document_type: str,
    dense_client: OllamaEmbeddingClient,
    sparse_embedder: SparseEmbedder,
    store: QdrantStore,
    chunker: GenericChunker,
) -> list[Chunk]:
    parser = DocxParser(path)
    sections = parser.extract_sections()

    chunks = chunker.chunk_docx(
        sections=sections,
        document_id=f"{document_type}_{path.stem}",
        document_title=parser.guess_title(sections),
        document_type=document_type,
        version="unknown",
        publication_date="unknown",
        source_file=str(path),
    )

    dense_embeddings = await dense_client.embed([c.text for c in chunks])
    sparse_embeddings = sparse_embedder.embed([c.text for c in chunks])
    await store.upsert(chunks, dense_embeddings, sparse_embeddings)
    return chunks


async def main() -> None:
    settings = get_settings()
    documents_dir = settings.data_dir / "documents"

    dense_client = OllamaEmbeddingClient(base_url=settings.ollama_base_url, model=settings.embedding_model)
    sparse_embedder = SparseEmbedder()
    store = QdrantStore(url=settings.qdrant_url, collection_name=settings.qdrant_collection)
    standard_chunker = StructureAwareChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    generic_chunker = GenericChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)

    await store.ensure_collection()

    total_documents = 0
    total_chunks = 0

    pci_dss_paths = sorted((documents_dir / "pci_dss").glob("*.pdf"))
    for path in pci_dss_paths:
        print(f"Ingesting {path} (pci_dss) ...")
        chunks = await ingest_pdf(path, dense_client, sparse_embedder, store, standard_chunker)
        print(f"  {len(chunks)} chunks")
        total_documents += 1
        total_chunks += len(chunks)

    for document_type in ("faq", "guidance"):
        for path in sorted((documents_dir / document_type).glob("*.pdf")):
            print(f"Ingesting {path} ({document_type}) ...")
            chunks = await ingest_generic_pdf(path, document_type, dense_client, sparse_embedder, store, generic_chunker)
            print(f"  {len(chunks)} chunks")
            total_documents += 1
            total_chunks += len(chunks)

    # `data/documents/org/` holds example CLIENT evidence documents - see
    # the module docstring for why they're deliberately not indexed here.

    print(f"Done: {total_documents} document(s), {total_chunks} chunks indexed into {settings.qdrant_collection!r}.")


if __name__ == "__main__":
    asyncio.run(main())
