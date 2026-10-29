"""Create tables, then apply the Postgres-only indexes in db/schema.sql."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sqlalchemy import text  # noqa: E402
from app.config import is_postgres  # noqa: E402
from app.db import engine, init_db  # noqa: E402

init_db()
print("tables ready")

if is_postgres():
    sql = pathlib.Path(__file__).resolve().parents[1] / "db" / "schema.sql"
    with engine.begin() as conn:
        for statement in [s.strip() for s in sql.read_text().split(";") if s.strip()]:
            conn.execute(text(statement))
    print("pgvector extension + indexes applied")
else:
    print("sqlite dev database - skipping pgvector indexes")
