"""Data model.

Retention rule of thumb encoded here: anything that identifies a student lives
on `users` / `resumes` and is deletable in one cascade. Everything the model
learns from (`outcomes`, `skill_weights`) is de-identified so a deletion
request never destroys the research dataset.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    JSON,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .config import settings
from .db import Base, Vector


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Institution(Base):
    __tablename__ = "institutions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(200))
    domain: Mapped[str] = mapped_column(String(120), unique=True)
    batch_year: Mapped[int] = mapped_column(Integer, default=2027)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    users: Mapped[list["User"]] = relationship(back_populates="institution")


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    full_name: Mapped[str] = mapped_column(String(160), default="")
    # student | placement_officer | admin
    role: Mapped[str] = mapped_column(String(32), default="student")
    branch: Mapped[str] = mapped_column(String(80), default="")
    grad_year: Mapped[int] = mapped_column(Integer, default=2027)
    institution_id: Mapped[str | None] = mapped_column(ForeignKey("institutions.id"), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    institution: Mapped[Institution | None] = relationship(back_populates="users")
    consents: Mapped[list["Consent"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    resumes: Mapped[list["Resume"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    reports: Mapped[list["Report"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class Consent(Base):
    """Append-only consent ledger. Granting and withdrawing are both rows, so
    there is always a provable history of what was agreed to and when."""

    __tablename__ = "consents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    # resume_processing | institutional_sharing | research_use
    scope: Mapped[str] = mapped_column(String(48))
    granted: Mapped[bool] = mapped_column(Boolean, default=True)
    policy_version: Mapped[str] = mapped_column(String(32), default=settings.consent_version)
    ip_hash: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    user: Mapped[User] = relationship(back_populates="consents")


class Resume(Base):
    __tablename__ = "resumes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    filename: Mapped[str] = mapped_column(String(255), default="pasted.txt")
    source: Mapped[str] = mapped_column(String(16), default="paste")  # pdf | docx | txt | paste
    text: Mapped[str] = mapped_column(Text, default="")
    # Structural signals from the parser, used by the ATS checker.
    parse_meta: Mapped[dict] = mapped_column(JSON, default=dict)
    purge_after: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    user: Mapped[User] = relationship(back_populates="resumes")


class JobDescription(Base):
    __tablename__ = "job_descriptions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    # JDs are public postings, not personal data — deduped by hash and shared.
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    role_title: Mapped[str] = mapped_column(String(200), default="")
    company: Mapped[str] = mapped_column(String(160), default="")
    seniority: Mapped[str] = mapped_column(String(64), default="")
    text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    resume_id: Mapped[str] = mapped_column(ForeignKey("resumes.id", ondelete="CASCADE"))
    jd_id: Mapped[str] = mapped_column(ForeignKey("job_descriptions.id"))
    match_score: Mapped[float] = mapped_column(Float, default=0.0)
    readiness: Mapped[str] = mapped_column(String(16), default="Not yet")
    headline: Mapped[str] = mapped_column(Text, default="")
    # Full rendered report: requirements, roadmap, rewrites, ats, questions.
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    engine_version: Mapped[str] = mapped_column(String(32), default="v1")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    user: Mapped[User] = relationship(back_populates="reports")
    outcome: Mapped["Outcome | None"] = relationship(
        back_populates="report", uselist=False, cascade="all, delete-orphan"
    )


class Embedding(Base):
    """The semantic index. One row per chunk of a resume or JD, so matching is
    a vector search rather than one whole-document comparison."""

    __tablename__ = "embeddings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    owner_type: Mapped[str] = mapped_column(String(16), index=True)  # resume | jd
    owner_id: Mapped[str] = mapped_column(String(36), index=True)
    chunk: Mapped[str] = mapped_column(Text)
    # skill | responsibility | achievement | education | requirement
    kind: Mapped[str] = mapped_column(String(32), default="chunk")
    vector: Mapped[list[float]] = mapped_column(Vector(settings.embedding_dim))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


Index("ix_embeddings_owner", Embedding.owner_type, Embedding.owner_id)


class Outcome(Base):
    """The feedback loop. De-identified on purpose: no user_id, so it survives
    an account deletion and stays usable as research data."""

    __tablename__ = "outcomes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    report_id: Mapped[str] = mapped_column(
        ForeignKey("reports.id", ondelete="CASCADE"), unique=True, index=True
    )
    # applied | screened | interview | offer | rejected | ghosted
    status: Mapped[str] = mapped_column(String(16))
    score_at_apply: Mapped[float] = mapped_column(Float, default=0.0)
    # Snapshot of requirement statuses so calibration survives report deletion.
    requirement_snapshot: Mapped[list] = mapped_column(JSON, default=list)
    days_to_response: Mapped[int | None] = mapped_column(Integer, nullable=True)
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    report: Mapped[Report] = relationship(back_populates="outcome")


class SkillWeight(Base):
    """Learned per-skill importance, recomputed from outcomes. This is what
    turns the match score from a guess into a prediction."""

    __tablename__ = "skill_weights"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    skill: Mapped[str] = mapped_column(String(160), index=True)
    role_family: Mapped[str] = mapped_column(String(80), default="general", index=True)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    # How much a callback rate moves when this requirement is proven vs missing.
    lift: Mapped[float] = mapped_column(Float, default=0.0)
    sample_size: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class AuditLog(Base):
    """Who touched student data, and why. Required before any college pilot."""

    __tablename__ = "audit_log"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    actor_id: Mapped[str] = mapped_column(String(36), index=True)
    action: Mapped[str] = mapped_column(String(64))
    target: Mapped[str] = mapped_column(String(120), default="")
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
