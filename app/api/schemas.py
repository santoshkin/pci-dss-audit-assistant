"""Pydantic request/response models for the Client Workspace API."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.db.models import ComplianceStatus, EvidenceType, FindingStatus, ProjectStatus


class KnowledgeDocumentType(str, enum.Enum):
    """Not a Postgres enum like EvidenceType - the knowledge base has no
    tracking table (PLAN.md section 7 decision #1: the requirement catalog
    lives in Qdrant, not duplicated in Postgres), so this exists purely
    for API input validation, matching the document_type values
    app/ingest.py's CLI already uses."""

    PCI_DSS = "pci_dss"
    FAQ = "faq"
    GUIDANCE = "guidance"
    SUPPLEMENTS = "supplements"
    OTHER = "other"


class ClientCreate(BaseModel):
    name: str


class ClientOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    qdrant_collection_name: str
    created_at: datetime


class ProjectCreate(BaseModel):
    name: str


class ProjectOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    client_id: uuid.UUID
    name: str
    status: ProjectStatus
    created_at: datetime


class RequirementStatusSet(BaseModel):
    status: ComplianceStatus
    notes: str | None = None


class RequirementStatusOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    project_id: uuid.UUID
    requirement_id: str
    status: ComplianceStatus
    notes: str | None
    updated_at: datetime


class FindingCreate(BaseModel):
    requirement_id: str
    description: str
    recommendation: str | None = None
    created_by: uuid.UUID | None = None


class FindingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    project_id: uuid.UUID
    requirement_id: str
    description: str
    recommendation: str | None
    status: FindingStatus
    created_by: uuid.UUID | None
    created_at: datetime


class EvidenceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    project_id: uuid.UUID
    filename: str
    evidence_type: EvidenceType
    qdrant_document_id: str
    chunk_count: int
    requirement_id: str | None
    suggested_requirement_id: str | None
    created_at: datetime


class EvidenceRequirementSet(BaseModel):
    requirement_id: str


class ProjectChatRequest(BaseModel):
    question: str


class ProjectChatResponse(BaseModel):
    answer: str


class KnowledgeDocumentUploadOut(BaseModel):
    filename: str
    document_type: KnowledgeDocumentType
    document_id: str
    document_title: str
    chunk_count: int


class FindingUpdate(BaseModel):
    status: FindingStatus | None = None
    description: str | None = None
    recommendation: str | None = None
