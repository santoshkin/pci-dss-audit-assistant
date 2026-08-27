"""Async SQLAlchemy engine/session for the Client Workspace's structured
data (PLAN.md section 2: Client, AuditProject, RequirementStatus, Finding,
User). Kept separate from Qdrant, which holds text/embeddings only - see
ARCHITECTURE.md section 1 ("обновлённая архитектура") for why the workspace
needs a relational store at all.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings


class Base(DeclarativeBase):
    pass


_engine = create_async_engine(get_settings().postgres_dsn)
session_factory = async_sessionmaker(_engine, expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with session_factory() as session:
        yield session
