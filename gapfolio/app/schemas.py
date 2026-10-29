from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, EmailStr, Field


# --- auth -----------------------------------------------------------------

class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    full_name: str = ""
    role: Literal["student", "placement_officer"] = "student"
    branch: str = ""
    grad_year: int = 2027
    institution_domain: str | None = None
    invite_code: str = ""


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    role: str
    consent_required: list[str] = []


# --- consent --------------------------------------------------------------

class ConsentIn(BaseModel):
    scope: Literal["resume_processing", "institutional_sharing", "research_use"]
    granted: bool = True


class ConsentOut(BaseModel):
    scope: str
    granted: bool
    policy_version: str
    created_at: datetime

    class Config:
        from_attributes = True


# --- documents ------------------------------------------------------------

class ResumeOut(BaseModel):
    id: str
    filename: str
    source: str
    char_count: int
    warnings: list[dict] = []
    parse_confidence: float | None = None
    parse_confidence_level: str | None = None
    parse_confidence_notes: list[str] = []
    skills_detected: list[str] = []
    purge_after: datetime | None = None


class PasteIn(BaseModel):
    text: str = Field(min_length=100)
    filename: str = "pasted.txt"


# --- analysis -------------------------------------------------------------

class AnalyzeIn(BaseModel):
    resume_id: str
    jd_text: str = Field(min_length=80)
    company: str = ""


class RequirementOut(BaseModel):
    label: str
    jd_quote: str
    weight: str
    status: str
    evidence: str | None
    similarity: float


class ReportOut(BaseModel):
    id: str
    role_title: str
    match_score: float
    readiness: str
    headline: str
    requirements: list[RequirementOut]
    roadmap: list[dict]
    rewrites: list[dict]
    questions: list[str]
    ats: list[dict]
    parse_confidence: dict | None = None
    skills_detected: list[str] = []
    engine_version: str
    created_at: datetime


class ReportSummary(BaseModel):
    id: str
    role_title: str
    match_score: float
    readiness: str
    created_at: datetime
    outcome: str | None = None


# --- outcomes -------------------------------------------------------------

class OutcomeIn(BaseModel):
    status: Literal["applied", "screened", "interview", "offer", "rejected", "ghosted"]
    days_to_response: int | None = None
    note: str = ""


# --- progress -------------------------------------------------------------

class SkillTrend(BaseModel):
    label: str
    first_status: str
    latest_status: str
    closed: bool


class ProgressOut(BaseModel):
    reports: int
    first_score: float | None
    latest_score: float | None
    delta: float | None
    callbacks: int
    applications: int
    callback_rate: float | None
    closed_gaps: list[str]
    open_gaps: list[str]
    timeline: list[dict[str, Any]]
