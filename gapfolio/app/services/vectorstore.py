"""Vector index.

On Postgres this is a pgvector cosine search that can use an HNSW index. On
SQLite it is the same query in numpy. Callers get the same `Hit` list either
way, so the dev laptop and the deployed service behave identically apart from
speed.
"""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import delete, select, text as sql_text
from sqlalchemy.orm import Session

from ..config import is_postgres
from ..models import Embedding
from .embeddings import chunk_jd, chunk_resume, cosine, embed


@dataclass
class Hit:
    chunk: str
    kind: str
    score: float


def index_document(db: Session, owner_type: str, owner_id: str, text: str) -> int:
    """(Re)build the index for one resume or JD. Idempotent."""
    db.execute(
        delete(Embedding).where(
            Embedding.owner_type == owner_type, Embedding.owner_id == owner_id
        )
    )
    pairs = chunk_resume(text) if owner_type == "resume" else chunk_jd(text)
    if not pairs:
        db.commit()
        return 0

    vectors = embed([c for c, _ in pairs])
    for (chunk, kind), vec in zip(pairs, vectors):
        db.add(
            Embedding(
                owner_type=owner_type, owner_id=owner_id, chunk=chunk, kind=kind, vector=vec
            )
        )
    db.commit()
    return len(pairs)


def search(
    db: Session, query_vec: list[float], owner_type: str, owner_id: str, k: int = 5
) -> list[Hit]:
    if is_postgres():
        rows = db.execute(
            sql_text(
                """
                SELECT chunk, kind, 1 - (vector <=> CAST(:q AS vector)) AS score
                FROM embeddings
                WHERE owner_type = :ot AND owner_id = :oid
                ORDER BY vector <=> CAST(:q AS vector)
                LIMIT :k
                """
            ),
            {"q": str(query_vec), "ot": owner_type, "oid": owner_id, "k": k},
        ).all()
        return [Hit(chunk=r[0], kind=r[1], score=float(r[2])) for r in rows]

    rows = db.scalars(
        select(Embedding).where(
            Embedding.owner_type == owner_type, Embedding.owner_id == owner_id
        )
    ).all()
    scored = [Hit(chunk=r.chunk, kind=r.kind, score=cosine(query_vec, r.vector)) for r in rows]
    scored.sort(key=lambda h: h.score, reverse=True)
    return scored[:k]


def drop_document(db: Session, owner_type: str, owner_id: str) -> None:
    db.execute(
        delete(Embedding).where(
            Embedding.owner_type == owner_type, Embedding.owner_id == owner_id
        )
    )
    db.commit()
