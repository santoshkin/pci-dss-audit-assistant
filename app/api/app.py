"""Client Workspace API (PLAN.md section 3): create a project, link
findings to a requirement, mark compliance status (Phase 1); upload and
link client evidence artefacts (Phase 2); ask questions in a project's
context, searching that client's evidence collection alongside the shared
Standard one (PLAN.md section 3 Phase 1 note "работа в контексте
клиента", deferred until Evidence intake gave each client an actual
collection to search - see `app/services/project_chat.py`); upload
documents into the shared knowledge base, and full CRUD (get/delete,
finding status updates, pagination) so other services can integrate
against this API directly instead of only the CLI entry points
(ARCHITECTURE.md section 22). Also mounts `app/web`'s minimal server-
rendered UI at /ui (PLAN.md Phase 4, ARCHITECTURE.md section 23) - open
http://<host>/ui/clients in a browser. Run with:

    uvicorn app.api.app:app --reload

No auth on this API by design (per user decision, ARCHITECTURE.md section
22) - it's meant to run in a closed network alongside the services that
call it, not to be exposed publicly.

`app/chat.py`'s own CLI entry point remains the way to ask
Standard-only questions with no client/project context.
"""

from __future__ import annotations

import uuid

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import (
    get_dense_client,
    get_generation_client,
    get_generic_chunker,
    get_reranker,
    get_sparse_embedder,
    get_standard_chunker,
    get_standard_store,
)
from app.api.schemas import (
    ClientCreate,
    ClientOut,
    EvidenceOut,
    EvidenceRequirementSet,
    FindingCreate,
    FindingOut,
    FindingUpdate,
    KnowledgeDocumentType,
    KnowledgeDocumentUploadOut,
    ProjectChatRequest,
    ProjectChatResponse,
    ProjectCreate,
    ProjectOut,
    RequirementStatusOut,
    RequirementStatusSet,
)
from app.chunking import GenericChunker, StructureAwareChunker
from app.config import get_settings
from app.db import get_session
from app.db.models import EvidenceType
from app.embeddings import OllamaEmbeddingClient
from app.llm import OllamaGenerationClient
from app.reranking import LazyReranker
from app.retrieval import QdrantStore, SparseEmbedder
from app.services import (
    ClientWorkspaceService,
    EvidenceIntakeService,
    KnowledgeBaseIngestionService,
    ProjectChatService,
)
from app.web import router as web_router

app = FastAPI(title="PCI DSS Audit Assistant - Client Workspace API")
app.include_router(web_router)


def _service(session: AsyncSession = Depends(get_session)) -> ClientWorkspaceService:
    return ClientWorkspaceService(session)


def _evidence_service(
    session: AsyncSession = Depends(get_session),
    dense_client: OllamaEmbeddingClient = Depends(get_dense_client),
    sparse_embedder: SparseEmbedder = Depends(get_sparse_embedder),
    chunker: GenericChunker = Depends(get_generic_chunker),
    standard_store: QdrantStore = Depends(get_standard_store),
) -> EvidenceIntakeService:
    return EvidenceIntakeService(
        session, dense_client, sparse_embedder, chunker, standard_store, get_settings().qdrant_url
    )


def _project_chat_service(
    session: AsyncSession = Depends(get_session),
    dense_client: OllamaEmbeddingClient = Depends(get_dense_client),
    sparse_embedder: SparseEmbedder = Depends(get_sparse_embedder),
    reranker: LazyReranker = Depends(get_reranker),
    gen_client: OllamaGenerationClient = Depends(get_generation_client),
    standard_store: QdrantStore = Depends(get_standard_store),
) -> ProjectChatService:
    settings = get_settings()
    return ProjectChatService(
        session,
        dense_client,
        sparse_embedder,
        reranker,
        gen_client,
        standard_store,
        settings.qdrant_url,
        settings.top_k,
        settings.rerank_top_k,
    )


def _kb_service(
    dense_client: OllamaEmbeddingClient = Depends(get_dense_client),
    sparse_embedder: SparseEmbedder = Depends(get_sparse_embedder),
    standard_chunker: StructureAwareChunker = Depends(get_standard_chunker),
    generic_chunker: GenericChunker = Depends(get_generic_chunker),
    standard_store: QdrantStore = Depends(get_standard_store),
) -> KnowledgeBaseIngestionService:
    return KnowledgeBaseIngestionService(
        dense_client, sparse_embedder, standard_chunker, generic_chunker, standard_store
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/clients", response_model=ClientOut)
async def create_client(payload: ClientCreate, service: ClientWorkspaceService = Depends(_service)) -> ClientOut:
    client = await service.create_client(payload.name)
    return ClientOut.model_validate(client)


@app.get("/clients", response_model=list[ClientOut])
async def list_clients(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    service: ClientWorkspaceService = Depends(_service),
) -> list[ClientOut]:
    clients = await service.list_clients(limit=limit, offset=offset)
    return [ClientOut.model_validate(c) for c in clients]


@app.get("/clients/{client_id}", response_model=ClientOut)
async def get_client(client_id: uuid.UUID, service: ClientWorkspaceService = Depends(_service)) -> ClientOut:
    client = await service.get_client(client_id)
    if client is None:
        raise HTTPException(status_code=404, detail="Client not found")
    return ClientOut.model_validate(client)


@app.delete("/clients/{client_id}")
async def delete_client(client_id: uuid.UUID, service: ClientWorkspaceService = Depends(_service)) -> dict[str, str]:
    client = await service.get_client(client_id)
    if client is None:
        raise HTTPException(status_code=404, detail="Client not found")
    collection_name = client.qdrant_collection_name
    await service.delete_client(client_id)

    # Postgres cascade handles projects/findings/requirement statuses/
    # evidence rows; the client's whole Qdrant collection is separate
    # storage and has to be torn down here explicitly.
    store = QdrantStore(url=get_settings().qdrant_url, collection_name=collection_name)
    if await store.client.collection_exists(collection_name):
        await store.client.delete_collection(collection_name)
    return {"status": "deleted"}


@app.post("/clients/{client_id}/projects", response_model=ProjectOut)
async def create_project(
    client_id: uuid.UUID, payload: ProjectCreate, service: ClientWorkspaceService = Depends(_service)
) -> ProjectOut:
    project = await service.create_project(client_id, payload.name)
    if project is None:
        raise HTTPException(status_code=404, detail="Client not found")
    return ProjectOut.model_validate(project)


@app.get("/clients/{client_id}/projects", response_model=list[ProjectOut])
async def list_projects(
    client_id: uuid.UUID,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    service: ClientWorkspaceService = Depends(_service),
) -> list[ProjectOut]:
    if await service.get_client(client_id) is None:
        raise HTTPException(status_code=404, detail="Client not found")
    projects = await service.list_projects(client_id, limit=limit, offset=offset)
    return [ProjectOut.model_validate(p) for p in projects]


@app.get("/projects/{project_id}", response_model=ProjectOut)
async def get_project(project_id: uuid.UUID, service: ClientWorkspaceService = Depends(_service)) -> ProjectOut:
    project = await service.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return ProjectOut.model_validate(project)


@app.delete("/projects/{project_id}")
async def delete_project(
    project_id: uuid.UUID,
    service: ClientWorkspaceService = Depends(_service),
    evidence_service: EvidenceIntakeService = Depends(_evidence_service),
) -> dict[str, str]:
    if await service.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")

    # Delete each evidence's Qdrant chunks first (EvidenceIntakeService
    # knows how to reach the client's collection; ClientWorkspaceService
    # doesn't) - Postgres cascade alone would drop the rows but strand the
    # chunks as orphaned points.
    for evidence in await evidence_service.list_evidence(project_id) or []:
        await evidence_service.delete_evidence(project_id, evidence.id)

    await service.delete_project(project_id)
    return {"status": "deleted"}


@app.get("/projects/{project_id}/requirements", response_model=list[RequirementStatusOut])
async def list_requirement_statuses(
    project_id: uuid.UUID, service: ClientWorkspaceService = Depends(_service)
) -> list[RequirementStatusOut]:
    if await service.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    statuses = await service.list_requirement_statuses(project_id)
    return [RequirementStatusOut.model_validate(s) for s in statuses]


@app.put("/projects/{project_id}/requirements/{requirement_id}", response_model=RequirementStatusOut)
async def set_requirement_status(
    project_id: uuid.UUID,
    requirement_id: str,
    payload: RequirementStatusSet,
    service: ClientWorkspaceService = Depends(_service),
) -> RequirementStatusOut:
    record = await service.set_requirement_status(project_id, requirement_id, payload.status, payload.notes)
    if record is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return RequirementStatusOut.model_validate(record)


@app.post("/projects/{project_id}/findings", response_model=FindingOut)
async def create_finding(
    project_id: uuid.UUID, payload: FindingCreate, service: ClientWorkspaceService = Depends(_service)
) -> FindingOut:
    finding = await service.create_finding(
        project_id, payload.requirement_id, payload.description, payload.recommendation, payload.created_by
    )
    if finding is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return FindingOut.model_validate(finding)


@app.get("/projects/{project_id}/findings", response_model=list[FindingOut])
async def list_findings(
    project_id: uuid.UUID,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    service: ClientWorkspaceService = Depends(_service),
) -> list[FindingOut]:
    if await service.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    findings = await service.list_findings(project_id, limit=limit, offset=offset)
    return [FindingOut.model_validate(f) for f in findings]


@app.get("/projects/{project_id}/findings/{finding_id}", response_model=FindingOut)
async def get_finding(
    project_id: uuid.UUID, finding_id: uuid.UUID, service: ClientWorkspaceService = Depends(_service)
) -> FindingOut:
    finding = await service.get_finding(project_id, finding_id)
    if finding is None:
        raise HTTPException(status_code=404, detail="Finding not found")
    return FindingOut.model_validate(finding)


@app.patch("/projects/{project_id}/findings/{finding_id}", response_model=FindingOut)
async def update_finding(
    project_id: uuid.UUID,
    finding_id: uuid.UUID,
    payload: FindingUpdate,
    service: ClientWorkspaceService = Depends(_service),
) -> FindingOut:
    finding = await service.update_finding(
        project_id, finding_id, payload.status, payload.description, payload.recommendation
    )
    if finding is None:
        raise HTTPException(status_code=404, detail="Finding not found")
    return FindingOut.model_validate(finding)


@app.delete("/projects/{project_id}/findings/{finding_id}")
async def delete_finding(
    project_id: uuid.UUID, finding_id: uuid.UUID, service: ClientWorkspaceService = Depends(_service)
) -> dict[str, str]:
    if not await service.delete_finding(project_id, finding_id):
        raise HTTPException(status_code=404, detail="Finding not found")
    return {"status": "deleted"}


@app.post("/projects/{project_id}/evidence", response_model=EvidenceOut)
async def upload_evidence(
    project_id: uuid.UUID,
    file: UploadFile = File(...),
    evidence_type: EvidenceType = Form(...),
    service: EvidenceIntakeService = Depends(_evidence_service),
) -> EvidenceOut:
    content = await file.read()
    try:
        evidence = await service.ingest(project_id, file.filename or "evidence", content, evidence_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if evidence is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return EvidenceOut.model_validate(evidence)


@app.get("/projects/{project_id}/evidence", response_model=list[EvidenceOut])
async def list_evidence(
    project_id: uuid.UUID,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    service: EvidenceIntakeService = Depends(_evidence_service),
) -> list[EvidenceOut]:
    evidence = await service.list_evidence(project_id, limit=limit, offset=offset)
    if evidence is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return [EvidenceOut.model_validate(e) for e in evidence]


@app.get("/projects/{project_id}/evidence/{evidence_id}", response_model=EvidenceOut)
async def get_evidence(
    project_id: uuid.UUID, evidence_id: uuid.UUID, service: EvidenceIntakeService = Depends(_evidence_service)
) -> EvidenceOut:
    evidence = await service.get_evidence(project_id, evidence_id)
    if evidence is None:
        raise HTTPException(status_code=404, detail="Evidence not found")
    return EvidenceOut.model_validate(evidence)


@app.delete("/projects/{project_id}/evidence/{evidence_id}")
async def delete_evidence(
    project_id: uuid.UUID, evidence_id: uuid.UUID, service: EvidenceIntakeService = Depends(_evidence_service)
) -> dict[str, str]:
    if not await service.delete_evidence(project_id, evidence_id):
        raise HTTPException(status_code=404, detail="Evidence not found")
    return {"status": "deleted"}


@app.put("/projects/{project_id}/evidence/{evidence_id}/requirement", response_model=EvidenceOut)
async def set_evidence_requirement(
    project_id: uuid.UUID,
    evidence_id: uuid.UUID,
    payload: EvidenceRequirementSet,
    service: EvidenceIntakeService = Depends(_evidence_service),
) -> EvidenceOut:
    evidence = await service.set_requirement_link(project_id, evidence_id, payload.requirement_id)
    if evidence is None:
        raise HTTPException(status_code=404, detail="Evidence not found")
    return EvidenceOut.model_validate(evidence)


@app.post("/projects/{project_id}/chat", response_model=ProjectChatResponse)
async def project_chat(
    project_id: uuid.UUID,
    payload: ProjectChatRequest,
    service: ProjectChatService = Depends(_project_chat_service),
) -> ProjectChatResponse:
    answer = await service.answer(project_id, payload.question)
    if answer is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return ProjectChatResponse(answer=answer)


@app.post("/documents", response_model=KnowledgeDocumentUploadOut)
async def upload_knowledge_document(
    file: UploadFile = File(...),
    document_type: KnowledgeDocumentType = Form(...),
    service: KnowledgeBaseIngestionService = Depends(_kb_service),
) -> KnowledgeDocumentUploadOut:
    content = await file.read()
    try:
        result = await service.ingest(file.filename or "document", content, document_type.value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return KnowledgeDocumentUploadOut(
        filename=result.filename,
        document_type=document_type,
        document_id=result.document_id,
        document_title=result.document_title,
        chunk_count=result.chunk_count,
    )
