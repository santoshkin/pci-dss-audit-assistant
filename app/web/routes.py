"""Minimal server-rendered web UI (PLAN.md Phase 4): "выбор проекта/
клиента, чат... без истории" over the same services the JSON API
(app/api/app.py) already uses - no separate business logic, this only
renders HTML instead of JSON. Plain HTML forms, no htmx/JS: this sandbox
had no outbound internet access to vendor or verify a CDN script, and a
live demo in front of colleagues shouldn't depend on one working anyway -
a full page reload per action is a fine tradeoff for "minimal".

No auth (per user decision, ARCHITECTURE.md section 22 - closed network)
and no chat history (per PLAN.md Phase 4 note) - each question is
independent, nothing is persisted across page loads.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import (
    get_dense_client,
    get_generation_client,
    get_generic_chunker,
    get_reranker,
    get_sparse_embedder,
    get_standard_store,
)
from app.config import get_settings
from app.db import get_session
from app.db.models import ComplianceStatus, EvidenceType, FindingStatus
from app.embeddings import OllamaEmbeddingClient
from app.llm import OllamaGenerationClient
from app.reranking import LazyReranker
from app.retrieval import QdrantStore, SparseEmbedder
from app.services import ClientWorkspaceService, EvidenceIntakeService, ProjectChatService

router = APIRouter(prefix="/ui")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _service(session: AsyncSession = Depends(get_session)) -> ClientWorkspaceService:
    return ClientWorkspaceService(session)


def _evidence_service(
    session: AsyncSession = Depends(get_session),
    dense_client: OllamaEmbeddingClient = Depends(get_dense_client),
    sparse_embedder: SparseEmbedder = Depends(get_sparse_embedder),
    chunker=Depends(get_generic_chunker),
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
        session, dense_client, sparse_embedder, reranker, gen_client, standard_store,
        settings.qdrant_url, settings.top_k, settings.rerank_top_k,
    )


@router.get("/")
async def index() -> RedirectResponse:
    return RedirectResponse(url="/ui/clients", status_code=303)


@router.get("/clients")
async def list_clients_page(request: Request, service: ClientWorkspaceService = Depends(_service)):
    clients = await service.list_clients()
    return templates.TemplateResponse(request, "clients.html", {"clients": clients})


@router.post("/clients")
async def create_client_page(name: str = Form(...), service: ClientWorkspaceService = Depends(_service)):
    client = await service.create_client(name)
    return RedirectResponse(url=f"/ui/clients/{client.id}", status_code=303)


async def _load_client_and_project(
    project_id: uuid.UUID, service: ClientWorkspaceService
) -> tuple:
    project = await service.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    client = await service.get_client(project.client_id)
    assert client is not None
    return client, project


@router.get("/clients/{client_id}")
async def client_detail_page(request: Request, client_id: uuid.UUID, service: ClientWorkspaceService = Depends(_service)):
    client = await service.get_client(client_id)
    if client is None:
        raise HTTPException(status_code=404, detail="Client not found")
    projects = await service.list_projects(client_id)
    return templates.TemplateResponse(request, "client_detail.html", {"client": client, "projects": projects})


@router.post("/clients/{client_id}/projects")
async def create_project_page(
    client_id: uuid.UUID, name: str = Form(...), service: ClientWorkspaceService = Depends(_service)
):
    project = await service.create_project(client_id, name)
    if project is None:
        raise HTTPException(status_code=404, detail="Client not found")
    return RedirectResponse(url=f"/ui/projects/{project.id}", status_code=303)


async def _project_context(project_id: uuid.UUID, service: ClientWorkspaceService, evidence_service: EvidenceIntakeService) -> dict:
    client, project = await _load_client_and_project(project_id, service)
    return {
        "client": client,
        "project": project,
        "evidence": await evidence_service.list_evidence(project_id) or [],
        "findings": await service.list_findings(project_id),
        "requirement_statuses": await service.list_requirement_statuses(project_id),
    }


@router.get("/projects/{project_id}")
async def project_detail_page(
    request: Request,
    project_id: uuid.UUID,
    service: ClientWorkspaceService = Depends(_service),
    evidence_service: EvidenceIntakeService = Depends(_evidence_service),
):
    context = await _project_context(project_id, service, evidence_service)
    return templates.TemplateResponse(request, "project_detail.html", context)


@router.post("/projects/{project_id}/chat")
async def project_chat_page(
    request: Request,
    project_id: uuid.UUID,
    question: str = Form(...),
    service: ClientWorkspaceService = Depends(_service),
    evidence_service: EvidenceIntakeService = Depends(_evidence_service),
    chat_service: ProjectChatService = Depends(_project_chat_service),
):
    answer = await chat_service.answer(project_id, question)
    context = await _project_context(project_id, service, evidence_service)
    context["question"] = question
    context["answer"] = answer
    return templates.TemplateResponse(request, "project_detail.html", context)


@router.post("/projects/{project_id}/evidence")
async def upload_evidence_page(
    project_id: uuid.UUID,
    file: UploadFile = File(...),
    evidence_type: EvidenceType = Form(...),
    service: EvidenceIntakeService = Depends(_evidence_service),
):
    content = await file.read()
    try:
        await service.ingest(project_id, file.filename or "evidence", content, evidence_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(url=f"/ui/projects/{project_id}", status_code=303)


@router.post("/projects/{project_id}/findings")
async def create_finding_page(
    project_id: uuid.UUID,
    requirement_id: str = Form(...),
    description: str = Form(...),
    recommendation: str = Form(""),
    service: ClientWorkspaceService = Depends(_service),
):
    await service.create_finding(project_id, requirement_id, description, recommendation or None)
    return RedirectResponse(url=f"/ui/projects/{project_id}", status_code=303)


@router.post("/projects/{project_id}/findings/{finding_id}/status")
async def update_finding_status_page(
    project_id: uuid.UUID,
    finding_id: uuid.UUID,
    status: FindingStatus = Form(...),
    service: ClientWorkspaceService = Depends(_service),
):
    await service.update_finding(project_id, finding_id, status=status)
    return RedirectResponse(url=f"/ui/projects/{project_id}", status_code=303)


@router.post("/projects/{project_id}/requirements")
async def set_requirement_status_page(
    project_id: uuid.UUID,
    requirement_id: str = Form(...),
    status: ComplianceStatus = Form(...),
    notes: str = Form(""),
    service: ClientWorkspaceService = Depends(_service),
):
    await service.set_requirement_status(project_id, requirement_id, status, notes or None)
    return RedirectResponse(url=f"/ui/projects/{project_id}", status_code=303)
