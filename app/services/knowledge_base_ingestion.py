"""Knowledge-base document ingestion via API: upload a new/updated PCI DSS
Standard, FAQ, or Guidance document into the shared Qdrant collection
without needing the `python -m app.ingest` CLI (app/ingest.py) - the
counterpart to `EvidenceIntakeService` for the shared collection instead
of a per-client one, for services that want to manage the knowledge base
programmatically.

Follows the same temp-file/original-filename split as
`EvidenceIntakeService._parse_and_chunk`: `app/ingest.py`'s own
`ingest_pdf`/`ingest_generic_pdf`/`ingest_docx` pass `source_file=str(path)`
because in CLI usage `path` already IS a meaningful real file location
(`data/documents/*`) - not true for a spooled upload, so this reimplements
the same parsing calls directly rather than reusing those functions, to
avoid leaking a temp file's throwaway name into chunk metadata.
"""

from __future__ import annotations

import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.chunking import Chunk, GenericChunker, StructureAwareChunker
from app.embeddings import OllamaEmbeddingClient
from app.ingestion import DocxParser, GenericPdfParser, PdfLayoutParser, StructureExtractor
from app.retrieval import QdrantStore, SparseEmbedder

_SUPPORTED_SUFFIXES = {".pdf", ".docx"}


@dataclass(frozen=True)
class KnowledgeDocumentIngestResult:
    filename: str
    document_type: str
    document_id: str
    document_title: str
    chunk_count: int


class KnowledgeBaseIngestionService:
    def __init__(
        self,
        dense_client: OllamaEmbeddingClient,
        sparse_embedder: SparseEmbedder,
        standard_chunker: StructureAwareChunker,
        generic_chunker: GenericChunker,
        standard_store: QdrantStore,
    ) -> None:
        self.dense_client = dense_client
        self.sparse_embedder = sparse_embedder
        self.standard_chunker = standard_chunker
        self.generic_chunker = generic_chunker
        self.standard_store = standard_store

    def _parse_and_chunk(self, path: Path, filename: str, document_type: str) -> list[Chunk]:
        suffix = path.suffix.lower()
        base_name = Path(filename).stem

        if document_type == "pci_dss" and suffix == ".pdf":
            # The coordinate-based Standard parser (ARCHITECTURE.md section
            # 8) - only meaningful for the official PDF template, so a
            # "pci_dss"-typed .docx upload deliberately falls through to
            # the generic .docx path below instead of erroring: lower
            # precision, but still usable, matching the project's existing
            # accuracy/speed tradeoff for non-Standard formats.
            parser = PdfLayoutParser(path)
            metadata = parser.extract_metadata()
            pages = list(parser.iter_pages())
            section_titles = parser.extract_section_titles()
            extraction = StructureExtractor().extract(pages, section_titles)
            return self.standard_chunker.chunk(
                document_id=f"pci_dss_v{metadata.version}",
                document_title=metadata.title,
                document_type="pci_dss",
                version=metadata.version,
                publication_date=metadata.date,
                source_file=filename,
                requirements=extraction.requirements,
                testing_procedures=extraction.testing_procedures,
                guidance_blocks=extraction.guidance_blocks,
            )
        if suffix == ".pdf":
            parser = GenericPdfParser(path)
            pages = parser.extract_pages()
            title = parser.guess_title(pages)
            if title == path.stem:
                title = base_name
            return self.generic_chunker.chunk_pdf(
                pages=pages,
                document_id=f"{document_type}_{base_name}_{uuid.uuid4().hex[:8]}",
                document_title=title,
                document_type=document_type,
                version=parser.guess_version(pages),
                publication_date=parser.guess_date(pages),
                source_file=filename,
            )
        if suffix == ".docx":
            docx_parser = DocxParser(path)
            sections = docx_parser.extract_sections()
            title = docx_parser.guess_title(sections)
            if title == path.stem:
                title = base_name
            return self.generic_chunker.chunk_docx(
                sections=sections,
                document_id=f"{document_type}_{base_name}_{uuid.uuid4().hex[:8]}",
                document_title=title,
                document_type=document_type,
                version="unknown",
                publication_date="unknown",
                source_file=filename,
            )
        raise ValueError(f"Unsupported document file type: {suffix!r}")

    async def ingest(self, filename: str, content: bytes, document_type: str) -> KnowledgeDocumentIngestResult:
        suffix = Path(filename).suffix.lower()
        if suffix not in _SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported document file type: {suffix!r}")

        await self.standard_store.ensure_collection()

        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(content)
            tmp.flush()
            chunks = self._parse_and_chunk(Path(tmp.name), filename, document_type)

        if chunks:
            dense_embeddings = await self.dense_client.embed([c.text for c in chunks])
            sparse_embeddings = self.sparse_embedder.embed([c.text for c in chunks])
            await self.standard_store.upsert(chunks, dense_embeddings, sparse_embeddings)
            # The requirement_id catalog just changed - see
            # app/retrieval/qdrant_store.py's docstring for why a stale
            # cache would hide the new IDs from retrieval expansion.
            self.standard_store.invalidate_requirement_ids_cache()

        first = chunks[0] if chunks else None
        return KnowledgeDocumentIngestResult(
            filename=filename,
            document_type=document_type,
            document_id=first.document_id if first else "",
            document_title=first.document_title if first else filename,
            chunk_count=len(chunks),
        )
