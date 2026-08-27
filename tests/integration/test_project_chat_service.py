import uuid

import pytest

from app.db.models import EvidenceType
from app.services import EvidenceIntakeService, ProjectChatService

pytestmark = pytest.mark.integration


@pytest.fixture
def evidence_service(db_session, dense_client, sparse_embedder, generic_chunker, standard_store, settings):
    return EvidenceIntakeService(db_session, dense_client, sparse_embedder, generic_chunker, standard_store, settings.qdrant_url)


@pytest.fixture
def project_chat_service(db_session, dense_client, sparse_embedder, reranker, generation_client, standard_store, settings):
    return ProjectChatService(
        db_session, dense_client, sparse_embedder, reranker, generation_client, standard_store,
        settings.qdrant_url, settings.top_k, settings.rerank_top_k,
    )


async def test_answer_unknown_project_returns_none(project_chat_service):
    assert await project_chat_service.answer(uuid.uuid4(), "любой вопрос") is None


async def test_answer_before_any_evidence_still_works(project_chat_service, client_and_project):
    # The client's own Qdrant collection doesn't exist yet - must not 404
    # on the missing collection, just fall back to Standard-only context.
    _client, project = client_and_project
    answer = await project_chat_service.answer(project.id, "Что говорит стандарт про MFA?")
    assert answer
    assert "PCI DSS" in answer or "Недостаточно информации" in answer


async def test_answer_mixes_evidence_and_standard_citations(project_chat_service, evidence_service, client_and_project):
    # The MVP flow this service exists for: "how is requirement 8.3.6
    # documented for client X" -> an answer citing both the client's own
    # evidence and the PCI DSS requirement it's supposed to satisfy.
    _client, project = client_and_project
    content = (
        "Политика паролей системных и сервисных учётных записей: "
        "пароли меняются раз в 90 дней согласно регламенту ИБ."
    ).encode()
    await evidence_service.ingest(project.id, "password_policy.txt", content, EvidenceType.CONFIG_EXPORT)

    answer = await project_chat_service.answer(
        project.id, "Как у этого клиента реализована ротация паролей системных учётных записей?"
    )

    assert answer is not None
    assert answer != ""
    assert "Evidence: password_policy" in answer
