# PCI DSS Audit Assistant

A retrieval-augmented assistant for a team of PCI DSS QSA auditors. It keeps two knowledge domains separate but joinable:

1. **The shared PCI DSS Standard** (+ FAQ, Guidance) — one authoritative corpus, reused across every client.
2. **Per-client evidence** — each client's own policies, technical configuration dumps, and interview transcripts, isolated in its own Qdrant collection and never mixed with another client's data or with the shared Standard.

An auditor asks a question in the context of a specific client project ("how is requirement 8.3.6 documented for client X?", "does the password policy match PCI DSS?"), and the system retrieves both the relevant Standard text and the client's own evidence, and produces a structured, citation-backed answer: what the requirement says, what the evidence shows, the gap between them, and what's needed to close it. Citations are always built from the metadata of chunks that were actually retrieved — the model is not allowed to invent requirement numbers, page numbers, or facts about a client that aren't in the retrieved evidence.

This is an assistant, not an auditor replacement: it never emits a final compliance verdict, and formal report generation (ROC/AOC) is intentionally out of scope for now (see "Not built yet" below).

## Status

| Phase | What it covers | Status |
|---|---|---|
| 0 | Core RAG over the PCI DSS Standard (hybrid dense+sparse retrieval, structure-aware chunking, exact requirement-ID lookup, reranking) | Done |
| 1 | Client workspace: clients, audit projects, findings, requirement compliance status (Postgres) | Done |
| 2 | Evidence intake: per-client Qdrant collections, `.pdf`/`.docx`/`.txt`/`.json`/`.csv`/`.zip`/`.tar.gz` (technical-assessment archives from Windows and Unix, with encoding auto-detection) | Done |
| — | Client-context chat: merges Standard + client evidence retrieval, with structural requirement-family expansion and evidence query-expansion to cover broad "does X comply" questions | Done |
| 3 | Report generation (ROC/AOC/Action Plan) | **Blocked / deferred** — official docx templates exist in `data/templates/roc_aoc/` (not committed here, see below) but the fill-engine isn't built; both templates use structurally different mechanisms (legacy Word form fields with non-unique names for the AOC forms, no fields at all for the full ROC) that need per-template design work |
| 4 | Minimal web UI | Done — plain server-rendered HTML forms (no JS framework), see `app/web/` |

## Architecture at a glance

- **LLM / embeddings**: Ollama, served locally (`OLLAMA_BASE_URL`) — `bge-m3` for embeddings, a configurable chat model for generation (`qwen3.5:9b` in dev; larger models are available on the team's shared GPU server for later use).
- **Vector store**: Qdrant — one collection for the shared Standard, one collection per client for evidence, native hybrid dense+sparse search (RRF fusion), no separate BM25 engine.
- **Reranker**: a local cross-encoder (`bge-reranker-v2-m3`), lazily loaded and unloaded to share GPU memory with other services.
- **Relational store**: Postgres (via SQLAlchemy 2.0 async + Alembic) for clients, projects, findings, requirement statuses, evidence records.
- **API**: FastAPI, no auth (intended to run inside a closed network alongside the other services that call it — not exposed publicly).
- **Ingestion**: a coordinate-based structural parser for the Standard PDF itself (high accuracy, low ingest frequency); lighter-weight generic parsers for FAQ/Guidance/evidence documents and archives (accuracy/speed balance, since evidence formats are arbitrary and high-volume).

## Project layout

```
app/
  ingestion/     PDF/docx/archive/plain-text parsers
  chunking/      structure-aware chunker (Standard) + generic chunker (everything else)
  embeddings/    Ollama embedding client
  retrieval/     QdrantStore (hybrid search, requirement-family lookup), sparse (BM25) embedder
  reranking/     lazy-loaded cross-encoder reranker
  llm/           Ollama generation client
  prompts/       system prompt
  db/            SQLAlchemy models (Client, AuditProject, RequirementStatus, Finding, Evidence, User)
  services/      business logic: ClientWorkspaceService, EvidenceIntakeService,
                 KnowledgeBaseIngestionService, ProjectChatService
  api/           FastAPI app, Pydantic schemas, DI wiring
  web/           minimal server-rendered UI (plain HTML forms)
  chat.py        Phase 0 CLI: `python -m app.chat "question"` (Standard-only, no client context)
  ingest.py      Phase 0 CLI: `python -m app.ingest` (bulk-ingests data/documents/ into the shared collection)
alembic/         Postgres migrations
tests/
  unit/          pure-function tests, no external services
  integration/   real Postgres/Qdrant/Ollama, no mocks (see tests/conftest.py)
data/
  documents/     source documents for the shared knowledge base + example client evidence (data/documents/org/)
  templates/     ROC/AOC templates (not committed - see .gitignore)
```

## Running it

1. Copy `.env.example` to `.env` and fill in real values (Postgres, Qdrant, Ollama connection info).
2. `pip install -r requirements.txt` (Python 3.12 required — see the comment at the top of `requirements.txt`).
3. `alembic upgrade head` to create the Postgres schema.
4. `python -m app.ingest` to index the PCI DSS Standard/FAQ/Guidance into the shared Qdrant collection.
5. `uvicorn app.api.app:app --reload` — API at `/docs` (Swagger UI), web UI at `/ui/clients`.

`python -m app.chat "question"` works standalone for Standard-only Q&A without the API/DB.

## API surface

Full interactive docs at `/docs` once the server is running. Summary:

- `POST/GET /clients`, `GET/DELETE /clients/{id}` — client CRUD (delete also removes the client's Qdrant collection)
- `POST/GET /clients/{id}/projects`, `GET/DELETE /projects/{id}` — audit project CRUD
- `GET /projects/{id}/requirements`, `PUT /projects/{id}/requirements/{requirement_id}` — compliance status per requirement
- `POST/GET /projects/{id}/findings`, `GET/PATCH/DELETE /projects/{id}/findings/{finding_id}` — findings, with status workflow (draft → reviewed → final)
- `POST/GET /projects/{id}/evidence`, `GET/DELETE /projects/{id}/evidence/{evidence_id}`, `PUT .../requirement` — evidence upload (with an auto-suggested, auditor-confirmable requirement link) and lifecycle (delete also purges the evidence's chunks from Qdrant)
- `POST /projects/{id}/chat` — the core feature: ask a question in a project's context
- `POST /documents` — upload a document into the shared knowledge base (Standard/FAQ/Guidance/Supplements)

All list endpoints support `limit`/`offset` pagination.

## Testing

```
pytest tests/                    # everything - real Postgres/Qdrant/Ollama, no mocks, ~2-3 minutes
pytest tests/unit -q             # fast, no external services
pytest -m integration tests/     # only the ones that hit real services
```

Integration tests create everything under a `pytest_`-prefixed name and clean it up after themselves (including a session-start/end sweep for anything a killed test run left behind).

## Known limitations

- No auth on the API — by design, for now (closed-network deployment). Add it before exposing this outside a trusted network.
- Retrieval for broad compliance questions ("does X comply with PCI DSS?") is meaningfully better than plain embedding search (structural requirement-family expansion + evidence query-expansion via the Standard's own clause text) but still bounded by what the first hybrid search pass surfaces — very abstract questions can still miss a relevant requirement family entirely. Decomposing such questions into structured sub-queries per PCI DSS section is a natural next improvement, not yet built.
- No chat history — each question in the web UI/API is independent.
- Report generation (ROC/AOC) is not implemented — see the Status table above.
- `app/web`'s UI has no automated tests yet (covered manually); if it grows, add `httpx`-based tests alongside `tests/integration/test_api.py`.

## A note for anyone extending this project

This repository intentionally does not include `ARCHITECTURE.md`, an internal working log of every design decision, tradeoff, and bug found/fixed session by session — it's useful history for the original author but not necessary (and partly not appropriate) for onboarding external collaborators. If something in this README seems underspecified, the code itself (docstrings throughout explain *why*, not just *what*) is the source of truth.
