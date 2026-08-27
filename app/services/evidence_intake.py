"""Evidence ingestion (PLAN.md section 3, Phase 2): parse/chunk/embed a
client artefact (interview transcript, customer document, config export)
into that client's own Qdrant collection - never the shared Standard one
(PLAN.md section 2, "Мультиклиентность и изоляция": one collection per
client, isolated from other clients and from the shared knowledge base).

Metadata omits requirement/testing_procedure by default. A lightweight
nearest-neighbor search against the shared Standard collection (the same
hybrid dense+sparse search `app/chat.py` already uses) produces a
*suggested* requirement link at ingest time; this deliberately does NOT
ask the generation LLM to guess an ID - the system prompt elsewhere
forbids inventing requirement references, and a retrieval hit is at least
grounded in an actual chunk. The auditor still has to confirm (or
override) the link via `set_requirement_link` before it's treated as real.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.chunking import Chunk, GenericChunker
from app.db.models import AuditProject, Client, Evidence, EvidenceType
from app.embeddings import OllamaEmbeddingClient
from app.ingestion import ArchiveParser, DocxParser, GenericPdfParser, PlainTextParser
from app.retrieval import QdrantStore, SparseEmbedder
from app.retrieval.sparse import SparseVector

_TEXT_SUFFIXES = {".txt", ".json", ".csv"}
_ARCHIVE_SUFFIXES = {".zip", ".tgz"}  # ".tar.gz" is handled separately - a double suffix Path.suffix can't see
_SUPPORTED_SUFFIXES = _TEXT_SUFFIXES | {".pdf", ".docx"} | _ARCHIVE_SUFFIXES


def _archive_format(filename: str) -> str | None:
    """Technical-assessment evidence archives (Windows .zip, Unix .tar.gz/
    .tgz - PLAN.md section 3 Phase 2) - returns "zip"/"tar", or None if
    `filename` isn't a recognized archive."""
    lower = filename.lower()
    if lower.endswith(".zip"):
        return "zip"
    if lower.endswith(".tar.gz") or lower.endswith(".tgz"):
        return "tar"
    return None


def _base_name(filename: str) -> str:
    # Path(...).stem only strips the LAST suffix, which would leave a
    # stray ".tar" in "DESKTOP-ASVDD.tar.gz" - strip the whole compound
    # suffix explicitly for that one case.
    if filename.lower().endswith(".tar.gz"):
        return filename[: -len(".tar.gz")]
    return Path(filename).stem


class EvidenceIntakeService:
    def __init__(
        self,
        session: AsyncSession,
        dense_client: OllamaEmbeddingClient,
        sparse_embedder: SparseEmbedder,
        chunker: GenericChunker,
        standard_store: QdrantStore,
        qdrant_url: str,
    ) -> None:
        self.session = session
        self.dense_client = dense_client
        self.sparse_embedder = sparse_embedder
        self.chunker = chunker
        self.standard_store = standard_store
        self.qdrant_url = qdrant_url

    async def _get_project_with_client(self, project_id: uuid.UUID) -> tuple[AuditProject, Client] | None:
        project = await self.session.get(AuditProject, project_id)
        if project is None:
            return None
        client = await self.session.get(Client, project.client_id)
        assert client is not None
        return project, client

    def _parse_and_chunk(self, path: Path, filename: str, document_id: str) -> list[Chunk]:
        # `path` is a spooled temp file (see `ingest`), so any parser's
        # filename-derived fallback title (`self.path.stem`) would be a
        # meaningless temp name, not the evidence's actual name - fall back
        # to the *original* upload's stem instead whenever that happens.
        suffix = path.suffix.lower()
        original_title = _base_name(filename)

        archive_format = _archive_format(filename)
        if archive_format is not None:
            members = ArchiveParser(path, archive_format).extract_members()
            return self.chunker.chunk_archive(
                members=members,
                document_id=document_id,
                document_title=original_title,
                document_type="evidence",
                version="unknown",
                publication_date="unknown",
                source_file=filename,
            )

        if suffix == ".pdf":
            parser = GenericPdfParser(path)
            pages = parser.extract_pages()
            title = parser.guess_title(pages)
            if title == path.stem:
                title = original_title
            return self.chunker.chunk_pdf(
                pages=pages,
                document_id=document_id,
                document_title=title,
                document_type="evidence",
                version="unknown",
                publication_date="unknown",
                source_file=filename,
            )
        if suffix == ".docx":
            docx_parser = DocxParser(path)
            sections = docx_parser.extract_sections()
            title = docx_parser.guess_title(sections)
            if title == path.stem:
                title = original_title
            return self.chunker.chunk_docx(
                sections=sections,
                document_id=document_id,
                document_title=title,
                document_type="evidence",
                version="unknown",
                publication_date="unknown",
                source_file=filename,
            )
        if suffix in _TEXT_SUFFIXES:
            text_parser = PlainTextParser(path)
            return self.chunker.chunk_text(
                text=text_parser.extract_text(),
                document_id=document_id,
                document_title=original_title,
                document_type="evidence",
                version="unknown",
                publication_date="unknown",
                source_file=filename,
            )
        raise ValueError(f"Unsupported evidence file type: {suffix!r}")

    async def _suggest_requirement_id(self, dense_vector: list[float], sparse_vector: SparseVector) -> str | None:
        results = await self.standard_store.search(dense_vector, sparse_vector, top_k=5)
        for record in results:
            requirement_id = record.payload.get("requirement_id")
            if requirement_id:
                return requirement_id
        return None

    async def ingest(
        self,
        project_id: uuid.UUID,
        filename: str,
        content: bytes,
        evidence_type: EvidenceType,
    ) -> Evidence | None:
        found = await self._get_project_with_client(project_id)
        if found is None:
            return None
        _project, client = found

        suffix = Path(filename).suffix.lower()
        if _archive_format(filename) is None and suffix not in _SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported evidence file type: {suffix!r}")

        document_id = f"evidence_{evidence_type.value}_{_base_name(filename)}_{uuid.uuid4().hex[:8]}"
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(content)
            tmp.flush()
            chunks = self._parse_and_chunk(Path(tmp.name), filename, document_id)

        suggested_requirement_id: str | None = None
        if chunks:
            dense_embeddings = await self.dense_client.embed([c.text for c in chunks])
            sparse_embeddings = self.sparse_embedder.embed([c.text for c in chunks])

            client_store = QdrantStore(url=self.qdrant_url, collection_name=client.qdrant_collection_name)
            await client_store.ensure_collection()
            await client_store.upsert(chunks, dense_embeddings, sparse_embeddings)

            suggested_requirement_id = await self._suggest_requirement_id(dense_embeddings[0], sparse_embeddings[0])

        evidence = Evidence(
            project_id=project_id,
            filename=filename,
            evidence_type=evidence_type,
            qdrant_document_id=document_id,
            chunk_count=len(chunks),
            suggested_requirement_id=suggested_requirement_id,
        )
        self.session.add(evidence)
        await self.session.commit()
        await self.session.refresh(evidence)
        return evidence

    async def list_evidence(self, project_id: uuid.UUID, limit: int = 100, offset: int = 0) -> list[Evidence] | None:
        if await self.session.get(AuditProject, project_id) is None:
            return None
        result = await self.session.execute(
            select(Evidence)
            .where(Evidence.project_id == project_id)
            .order_by(Evidence.created_at)
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def get_evidence(self, project_id: uuid.UUID, evidence_id: uuid.UUID) -> Evidence | None:
        evidence = await self.session.get(Evidence, evidence_id)
        if evidence is None or evidence.project_id != project_id:
            return None
        return evidence

    async def delete_evidence(self, project_id: uuid.UUID, evidence_id: uuid.UUID) -> bool:
        """Deletes both the Postgres row and its chunks in the client's
        Qdrant collection (matched by the evidence's own
        `qdrant_document_id`) - unlike a project/client delete, this one
        can clean up Qdrant itself since it already knows which client
        owns this evidence."""
        evidence = await self.get_evidence(project_id, evidence_id)
        if evidence is None:
            return False
        project = await self.session.get(AuditProject, project_id)
        client = await self.session.get(Client, project.client_id)
        client_store = QdrantStore(url=self.qdrant_url, collection_name=client.qdrant_collection_name)
        await client_store.delete_by_document_id(evidence.qdrant_document_id)
        await self.session.delete(evidence)
        await self.session.commit()
        return True

    async def set_requirement_link(
        self, project_id: uuid.UUID, evidence_id: uuid.UUID, requirement_id: str
    ) -> Evidence | None:
        evidence = await self.session.get(Evidence, evidence_id)
        if evidence is None or evidence.project_id != project_id:
            return None
        evidence.requirement_id = requirement_id
        await self.session.commit()
        await self.session.refresh(evidence)
        return evidence
