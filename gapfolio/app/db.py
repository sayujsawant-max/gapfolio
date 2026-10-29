"""Database wiring.

The vector column is the one place Postgres and SQLite genuinely differ, so it
is isolated in `Vector`: a real pgvector column on Postgres, a JSON-encoded
list on SQLite. Query code never sees the difference — it goes through
services/vectorstore.py.
"""
import json
from typing import Iterator

from sqlalchemy import create_engine, types
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import settings, is_postgres

connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Vector(types.TypeDecorator):
    """float[] column: pgvector `vector(dim)` on Postgres, TEXT on SQLite."""

    impl = types.Text
    cache_ok = True

    def __init__(self, dim: int = 1024):
        self.dim = dim
        super().__init__()

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            from pgvector.sqlalchemy import Vector as PGVector  # type: ignore

            return dialect.type_descriptor(PGVector(self.dim))
        return dialect.type_descriptor(types.Text())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        value = [float(x) for x in value]
        if dialect.name == "postgresql":
            return value
        return json.dumps(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return list(value)
        return json.loads(value)


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create tables. Postgres also needs the extension and ANN index, which
    live in db/schema.sql and are applied by scripts/migrate.py."""
    if is_postgres():
        from sqlalchemy import text

        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    from . import models  # noqa: F401  (registers mappers)

    Base.metadata.create_all(engine)
