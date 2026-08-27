"""Business logic for the Client Workspace (PLAN.md section 3, Phase 1:
create a project, link findings to a requirement, mark compliance status -
no report generation yet, that's Phase 3 and blocked on ROC/AOC templates
(ARCHITECTURE.md/PLAN.md section 7, decision 3).
"""

from __future__ import annotations

import re
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AuditProject, Client, ComplianceStatus, Finding, FindingStatus, RequirementStatus

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _collection_name_for(client_name: str) -> str:
    slug = _SLUG_RE.sub("_", client_name.lower()).strip("_")
    return f"client_{slug}_{uuid.uuid4().hex[:8]}"


class ClientWorkspaceService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    # -- Clients ----------------------------------------------------------

    async def create_client(self, name: str) -> Client:
        client = Client(name=name, qdrant_collection_name=_collection_name_for(name))
        self.session.add(client)
        await self.session.commit()
        await self.session.refresh(client)
        return client

    async def list_clients(self, limit: int = 100, offset: int = 0) -> list[Client]:
        result = await self.session.execute(
            select(Client).order_by(Client.created_at).limit(limit).offset(offset)
        )
        return list(result.scalars().all())

    async def get_client(self, client_id: uuid.UUID) -> Client | None:
        return await self.session.get(Client, client_id)

    async def delete_client(self, client_id: uuid.UUID) -> bool:
        """Postgres-only: cascades to projects/findings/requirement
        statuses/evidence rows via FK. Callers that also need the
        client's Qdrant collection removed (its evidence chunks aren't
        reachable from here) do that separately - see app/api/app.py."""
        client = await self.get_client(client_id)
        if client is None:
            return False
        await self.session.delete(client)
        await self.session.commit()
        return True

    # -- Audit projects -----------------------------------------------------

    async def create_project(self, client_id: uuid.UUID, name: str) -> AuditProject | None:
        if await self.get_client(client_id) is None:
            return None
        project = AuditProject(client_id=client_id, name=name)
        self.session.add(project)
        await self.session.commit()
        await self.session.refresh(project)
        return project

    async def get_project(self, project_id: uuid.UUID) -> AuditProject | None:
        return await self.session.get(AuditProject, project_id)

    async def list_projects(self, client_id: uuid.UUID, limit: int = 100, offset: int = 0) -> list[AuditProject]:
        result = await self.session.execute(
            select(AuditProject)
            .where(AuditProject.client_id == client_id)
            .order_by(AuditProject.created_at)
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def delete_project(self, project_id: uuid.UUID) -> bool:
        """Postgres-only, same caveat as delete_client: cascades findings/
        requirement statuses/evidence rows, but not their Qdrant chunks."""
        project = await self.get_project(project_id)
        if project is None:
            return False
        await self.session.delete(project)
        await self.session.commit()
        return True

    # -- Requirement status --------------------------------------------------

    async def set_requirement_status(
        self, project_id: uuid.UUID, requirement_id: str, status: ComplianceStatus, notes: str | None = None
    ) -> RequirementStatus | None:
        if await self.get_project(project_id) is None:
            return None

        result = await self.session.execute(
            select(RequirementStatus).where(
                RequirementStatus.project_id == project_id, RequirementStatus.requirement_id == requirement_id
            )
        )
        existing = result.scalar_one_or_none()
        if existing is not None:
            existing.status = status
            existing.notes = notes
            await self.session.commit()
            await self.session.refresh(existing)
            return existing

        record = RequirementStatus(project_id=project_id, requirement_id=requirement_id, status=status, notes=notes)
        self.session.add(record)
        await self.session.commit()
        await self.session.refresh(record)
        return record

    async def list_requirement_statuses(self, project_id: uuid.UUID) -> list[RequirementStatus]:
        result = await self.session.execute(
            select(RequirementStatus).where(RequirementStatus.project_id == project_id)
        )
        return list(result.scalars().all())

    # -- Findings ----------------------------------------------------------

    async def create_finding(
        self,
        project_id: uuid.UUID,
        requirement_id: str,
        description: str,
        recommendation: str | None = None,
        created_by: uuid.UUID | None = None,
    ) -> Finding | None:
        if await self.get_project(project_id) is None:
            return None
        finding = Finding(
            project_id=project_id,
            requirement_id=requirement_id,
            description=description,
            recommendation=recommendation,
            created_by=created_by,
        )
        self.session.add(finding)
        await self.session.commit()
        await self.session.refresh(finding)
        return finding

    async def list_findings(self, project_id: uuid.UUID, limit: int = 100, offset: int = 0) -> list[Finding]:
        result = await self.session.execute(
            select(Finding)
            .where(Finding.project_id == project_id)
            .order_by(Finding.created_at)
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def get_finding(self, project_id: uuid.UUID, finding_id: uuid.UUID) -> Finding | None:
        finding = await self.session.get(Finding, finding_id)
        if finding is None or finding.project_id != project_id:
            return None
        return finding

    async def update_finding(
        self,
        project_id: uuid.UUID,
        finding_id: uuid.UUID,
        status: FindingStatus | None = None,
        description: str | None = None,
        recommendation: str | None = None,
    ) -> Finding | None:
        finding = await self.get_finding(project_id, finding_id)
        if finding is None:
            return None
        if status is not None:
            finding.status = status
        if description is not None:
            finding.description = description
        if recommendation is not None:
            finding.recommendation = recommendation
        await self.session.commit()
        await self.session.refresh(finding)
        return finding

    async def delete_finding(self, project_id: uuid.UUID, finding_id: uuid.UUID) -> bool:
        finding = await self.get_finding(project_id, finding_id)
        if finding is None:
            return False
        await self.session.delete(finding)
        await self.session.commit()
        return True
