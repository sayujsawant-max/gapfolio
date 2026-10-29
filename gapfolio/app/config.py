"""Runtime configuration.

Everything is env-driven so the same image runs on a laptop (SQLite) and in
production (Postgres + pgvector) with no code changes.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- core -------------------------------------------------------
    app_name: str = "GapFolio API"
    env: str = "development"
    secret_key: str = "dev-secret-change-me"
    access_token_minutes: int = 60 * 12

    # postgresql+psycopg://gapfolio:gapfolio@localhost:5432/gapfolio
    database_url: str = "sqlite:///./gapfolio.db"

    # --- model providers --------------------------------------------
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"

    # sbert | voyage | openai | hash
    #   sbert = Sentence-Transformers model running locally, no key, no per-call cost
    #   hash  = deterministic offline stub, tests only
    embedding_provider: str = "sbert"
    embedding_api_key: str = ""
    embedding_model: str = "voyage-3"          # remote providers only
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384                   # 384 for the default SBERT model

    # Placement-officer accounts can only be created with this code. Empty
    # means they cannot be created through the public API at all.
    officer_invite_code: str = ""

    # --- public deployment ------------------------------------------
    # Serve the built React app from this API so one origin carries both.
    serve_ui: bool = True
    ui_dist: str = "../gapfolio-ui/dist"
    cors_origins: str = "http://localhost:3000,http://localhost:5173"
    # Requests per minute per client: general, and the expensive/abusable routes.
    rate_limit_general: int = 120
    rate_limit_heavy: int = 12

    # --- uploads ----------------------------------------------------
    max_upload_bytes: int = 5 * 1024 * 1024
    # Raw resume files are parsed in memory and never written to disk.
    store_raw_text: bool = True

    # --- privacy ----------------------------------------------------
    # Days a resume's text is retained after its last analysis.
    retention_days: int = 365
    consent_version: str = "2026-08-v1"

    class Config:
        env_file = ".env"
        env_prefix = "GAPFOLIO_"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


def is_postgres() -> bool:
    return settings.database_url.startswith("postgres")
