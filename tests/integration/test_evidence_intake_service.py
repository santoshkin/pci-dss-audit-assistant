import uuid

import pytest
from qdrant_client import models

from app.db.models import EvidenceType
from app.services import EvidenceIntakeService

pytestmark = pytest.mark.integration


@pytest.fixture
def evidence_service(db_session, dense_client, sparse_embedder, generic_chunker, standard_store, settings):
    return EvidenceIntakeService(db_session, dense_client, sparse_embedder, generic_chunker, standard_store, settings.qdrant_url)


async def test_ingest_unknown_project_returns_none(evidence_service):
    result = await evidence_service.ingest(uuid.uuid4(), "x.txt", b"content", EvidenceType.OTHER)
    assert result is None


async def test_ingest_unsupported_file_type_raises(evidence_service, client_and_project):
    _client, project = client_and_project
    with pytest.raises(ValueError):
        await evidence_service.ingest(project.id, "malware.exe", b"content", EvidenceType.OTHER)


async def test_ingest_txt_evidence_lands_only_in_client_collection(
    evidence_service, client_and_project, qdrant_client, standard_store
):
    client, project = client_and_project
    content = "Интервью: пароли администраторов CDE меняются каждые 60 дней.".encode()

    evidence = await evidence_service.ingest(project.id, "interview.txt", content, EvidenceType.INTERVIEW_TRANSCRIPT)

    assert evidence is not None
    assert evidence.chunk_count == 1
    # Regression: document_title must come from the original upload name,
    # not the spooled temp file's random name (ARCHITECTURE.md section 17).
    client_points, _ = await qdrant_client.scroll(
        collection_name=client.qdrant_collection_name,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=evidence.qdrant_document_id))]
        ),
        limit=10,
    )
    assert len(client_points) == 1
    assert client_points[0].payload["document_title"] == "interview"
    assert client_points[0].payload["document_type"] == "evidence"

    # Isolation: this evidence must never show up in the shared collection.
    shared_points, _ = await qdrant_client.scroll(
        collection_name=standard_store.collection_name,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=evidence.qdrant_document_id))]
        ),
        limit=10,
    )
    assert shared_points == []


async def test_ingest_json_evidence_pretty_prints_and_suggests_requirement(evidence_service, client_and_project):
    _client, project = client_and_project
    content = b'{"mfa_enabled": true, "rotation_days": 60}'

    evidence = await evidence_service.ingest(project.id, "config.json", content, EvidenceType.CONFIG_EXPORT)

    assert evidence.chunk_count == 1
    assert evidence.requirement_id is None  # not confirmed yet
    # Draft suggestion comes from a real hybrid search against the shared
    # Standard collection - just assert it's requirement-ID-shaped, not a
    # specific value (the actual nearest hit can shift as the corpus does).
    assert evidence.suggested_requirement_id is not None
    assert evidence.suggested_requirement_id.count(".") >= 1


async def test_list_evidence_unknown_project_returns_none(evidence_service):
    assert await evidence_service.list_evidence(uuid.uuid4()) is None


async def test_list_evidence_returns_uploaded_items(evidence_service, client_and_project):
    _client, project = client_and_project
    await evidence_service.ingest(project.id, "a.txt", b"first artefact", EvidenceType.OTHER)
    await evidence_service.ingest(project.id, "b.txt", b"second artefact", EvidenceType.OTHER)

    items = await evidence_service.list_evidence(project.id)
    assert {e.filename for e in items} == {"a.txt", "b.txt"}


async def test_set_requirement_link_confirms_without_overwriting_suggestion(evidence_service, client_and_project):
    _client, project = client_and_project
    evidence = await evidence_service.ingest(project.id, "interview.txt", b"MFA discussion", EvidenceType.INTERVIEW_TRANSCRIPT)
    suggested_before = evidence.suggested_requirement_id

    updated = await evidence_service.set_requirement_link(project.id, evidence.id, "8.4.2")

    assert updated.requirement_id == "8.4.2"
    assert updated.suggested_requirement_id == suggested_before  # untouched


async def test_set_requirement_link_unknown_evidence_returns_none(evidence_service, client_and_project):
    _client, project = client_and_project
    assert await evidence_service.set_requirement_link(project.id, uuid.uuid4(), "8.4.2") is None
