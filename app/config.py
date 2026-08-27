from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Ollama
    ollama_base_url: str
    ollama_model: str
    embedding_model: str
    # Ollama's own default is far below what most models actually support
    # (e.g. qwen3.5:9b: 262144) - explicit so retrieval-expanded contexts
    # (ARCHITECTURE.md section 21) don't silently truncate the answer.
    ollama_num_ctx: int = 16384

    # Reranker
    reranker_model: str
    reranker_idle_timeout_seconds: int
    reranker_device: str

    # Qdrant
    qdrant_url: str
    qdrant_collection: str

    # Postgres
    postgres_host: str
    postgres_port: int
    postgres_user: str
    postgres_password: str
    postgres_db: str

    # Retrieval
    top_k: int
    rerank_top_k: int
    min_retrieval_score: float

    # Chunking
    chunk_size: int
    chunk_overlap: int

    # Data
    data_dir: Path

    # Logging
    log_level: str = "INFO"

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
