"""ORM models for the Client Workspace (PLAN.md section 2: Client,
AuditProject, RequirementStatus, Finding, User from Phase 1; Evidence
added in Phase 2 - report entities are still Phase 3, not modeled yet).
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class ProjectStatus(str, enum.Enum):
    EVIDENCE_COLLECTION = "evidence_collection"
    ANALYSIS = "analysis"
    REPORT_DRAFT = "report_draft"
    REVIEW = "review"
    CLOSED = "closed"


class ComplianceStatus(str, enum.Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    NOT_APPLICABLE = "not_applicable"
    COMPENSATING_CONTROL = "compensating_control"
    IN_PROGRESS = "in_progress"


class FindingStatus(str, enum.Enum):
    DRAFT = "draft"
    REVIEWED = "reviewed"
    FINAL = "final"


class UserRole(str, enum.Enum):
    AUDITOR = "auditor"
    ADMIN = "admin"


class EvidenceType(str, enum.Enum):
    INTERVIEW_TRANSCRIPT = "interview_transcript"
    CUSTOMER_DOCUMENT = "customer_document"
    CONFIG_EXPORT = "config_export"
    OTHER = "other"


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


def _pg_enum(enum_cls: type[enum.Enum], name: str) -> Enum:
    # Without values_callable, SQLAlchemy stores the enum MEMBER NAME
    # ("NON_COMPLIANT") as the Postgres label, while the API's JSON
    # contract (and every other lowercase-string enum value in this
    # project) uses `.value` ("non_compliant") - this keeps raw SQL and
    # the API representation consistent instead of silently diverging.
    return Enum(enum_cls, name=name, values_callable=lambda e: [m.value for m in e])


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(String(255))
    role: Mapped[UserRole] = mapped_column(_pg_enum(UserRole, "user_role"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Client(Base):
    __tablename__ = "clients"

    id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(String(255), unique=True)
    # Isolation boundary (PLAN.md section 2, "Мультиклиентность и
    # изоляция" - one collection PER CLIENT, shared across that client's
    # audit projects): this client's evidence lives in its own Qdrant
    # collection, never the shared knowledge-base one or another client's.
    qdrant_collection_name: Mapped[str] = mapped_column(String(255), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    projects: Mapped[list["AuditProject"]] = relationship(back_populates="client", cascade="all, delete-orphan")


class AuditProject(Base):
    __tablename__ = "audit_projects"

    id: Mapped[uuid.UUID] = _uuid_pk()
    client_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("clients.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(255))
    status: Mapped[ProjectStatus] = mapped_column(
        _pg_enum(ProjectStatus, "project_status"), default=ProjectStatus.EVIDENCE_COLLECTION
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    client: Mapped[Client] = relationship(back_populates="projects")
    requirement_statuses: Mapped[list["RequirementStatus"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    findings: Mapped[list["Finding"]] = relationship(back_populates="project", cascade="all, delete-orphan")
    evidence: Mapped[list["Evidence"]] = relationship(back_populates="project", cascade="all, delete-orphan")


class RequirementStatus(Base):
    """Compliance status of one Requirement within one AuditProject.
    Requirement ID is a plain string, not a foreign key - the catalog of
    valid IDs lives in Qdrant (extracted from the Standard PDF, see
    ARCHITECTURE.md section 7/9, decision #1), not duplicated in Postgres.
    """

    __tablename__ = "requirement_statuses"
    __table_args__ = (UniqueConstraint("project_id", "requirement_id", name="uq_project_requirement"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("audit_projects.id", ondelete="CASCADE"))
    requirement_id: Mapped[str] = mapped_column(String(32))
    status: Mapped[ComplianceStatus] = mapped_column(_pg_enum(ComplianceStatus, "compliance_status"))
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    project: Mapped[AuditProject] = relationship(back_populates="requirement_statuses")


class Finding(Base):
    __tablename__ = "findings"

    id: Mapped[uuid.UUID] = _uuid_pk()
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("audit_projects.id", ondelete="CASCADE"))
    requirement_id: Mapped[str] = mapped_column(String(32))
    description: Mapped[str] = mapped_column(Text)
    recommendation: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[FindingStatus] = mapped_column(_pg_enum(FindingStatus, "finding_status"), default=FindingStatus.DRAFT)
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    project: Mapped[AuditProject] = relationship(back_populates="findings")


class Evidence(Base):
    """A client artefact (interview transcript, customer document, config
    export - PLAN.md section 3, Phase 2) chunked and indexed into its
    client's own Qdrant collection (`Client.qdrant_collection_name`), never
    the shared Standard one. `requirement_id` is the auditor-confirmed link
    to a PCI DSS requirement, left null until set explicitly;
    `suggested_requirement_id` is a non-binding draft produced at ingest
    time by a nearest-neighbor search against the shared Standard
    collection (see `app/services/evidence_intake.py`) - never trusted as
    the confirmed link on its own.
    """

    __tablename__ = "evidence"

    id: Mapped[uuid.UUID] = _uuid_pk()
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("audit_projects.id", ondelete="CASCADE"))
    filename: Mapped[str] = mapped_column(String(255))
    evidence_type: Mapped[EvidenceType] = mapped_column(_pg_enum(EvidenceType, "evidence_type"))
    qdrant_document_id: Mapped[str] = mapped_column(String(255))
    chunk_count: Mapped[int] = mapped_column(Integer)
    requirement_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    suggested_requirement_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    project: Mapped[AuditProject] = relationship(back_populates="evidence")
