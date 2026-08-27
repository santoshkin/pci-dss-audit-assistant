import uuid

import pytest

from app.db.models import ComplianceStatus
from app.services import ClientWorkspaceService

pytestmark = pytest.mark.integration


async def test_create_and_get_client(db_session, client_and_project):
    client, _project = client_and_project
    service = ClientWorkspaceService(db_session)
    fetched = await service.get_client(client.id)
    assert fetched is not None
    assert fetched.id == client.id
    assert fetched.qdrant_collection_name.startswith("client_")


async def test_get_unknown_client_returns_none(db_session):
    service = ClientWorkspaceService(db_session)
    assert await service.get_client(uuid.uuid4()) is None


async def test_create_project_for_unknown_client_returns_none(db_session):
    service = ClientWorkspaceService(db_session)
    assert await service.create_project(uuid.uuid4(), "orphan project") is None


async def test_set_requirement_status_upserts_not_duplicates(db_session, client_and_project):
    _client, project = client_and_project
    service = ClientWorkspaceService(db_session)

    first = await service.set_requirement_status(project.id, "8.4.2", ComplianceStatus.IN_PROGRESS, notes="draft")
    second = await service.set_requirement_status(project.id, "8.4.2", ComplianceStatus.COMPLIANT, notes="confirmed")

    assert first.id == second.id
    statuses = await service.list_requirement_statuses(project.id)
    assert len(statuses) == 1
    assert statuses[0].status == ComplianceStatus.COMPLIANT
    assert statuses[0].notes == "confirmed"


async def test_create_finding_for_unknown_project_returns_none(db_session):
    service = ClientWorkspaceService(db_session)
    assert await service.create_finding(uuid.uuid4(), "8.4.2", "gap description") is None


async def test_create_and_list_findings(db_session, client_and_project):
    _client, project = client_and_project
    service = ClientWorkspaceService(db_session)

    await service.create_finding(project.id, "8.4.2", "MFA not enforced for jumpbox access", "Enable MFA")
    findings = await service.list_findings(project.id)

    assert len(findings) == 1
    assert findings[0].requirement_id == "8.4.2"
    assert findings[0].recommendation == "Enable MFA"


async def test_client_name_is_unique(db_session, unique_name):
    service = ClientWorkspaceService(db_session)
    name = unique_name
    client = await service.create_client(name)
    client_id = client.id  # read before rollback expires the ORM object
    try:
        with pytest.raises(Exception):
            await service.create_client(name)
    finally:
        # A failed commit leaves the session unusable until rolled back.
        await db_session.rollback()
        from sqlalchemy import text

        await db_session.execute(text("DELETE FROM clients WHERE id = :id"), {"id": str(client_id)})
        await db_session.commit()
