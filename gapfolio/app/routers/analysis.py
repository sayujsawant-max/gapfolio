from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import JobDescription, Report, Resume, User
from ..schemas import AnalyzeIn, ReportOut, ReportSummary
from ..security import audit, require_consent
from ..services import matching, vectorstore
from ..services.parsing import ats_checks

router = APIRouter(prefix="/api", tags=["analysis"])


def _get_or_create_jd(db: Session, text: str, company: str) -> tuple[JobDescription, dict]:
    h = matching.jd_hash(text)
    jd = db.scalar(select(JobDescription).where(JobDescription.content_hash == h))
    extracted = matching.extract_requirements(text)
    if jd is None:
        jd = JobDescription(
            content_hash=h,
            text=text,
            company=company or extracted.get("company", ""),
            role_title=extracted.get("role_title", "")[:200],
            seniority=extracted.get("seniority", "")[:64],
        )
        db.add(jd)
        db.commit()
        vectorstore.index_document(db, "jd", jd.id, text)
    return jd, extracted


@router.post("/analyze", response_model=ReportOut, status_code=201)
def analyze(
    body: AnalyzeIn,
    user: User = Depends(require_consent("resume_processing")),
    db: Session = Depends(get_db),
):
    resume = db.get(Resume, body.resume_id)
    if not resume or resume.user_id != user.id:
        raise HTTPException(404, "No resume with that id on your account.")

    jd, extracted = _get_or_create_jd(db, body.jd_text, body.company)
    role_family = extracted.get("role_family") or "general"

    matches = matching.match_requirements(db, resume.id, extracted["requirements"])
    weights = matching.load_weights(db, role_family)
    score, readiness = matching.score(matches, weights)
    role = jd.role_title or "this role"
    head = matching.headline(matches, role)

    meta = resume.parse_meta or {}
    # The roadmap generator is given ranked gaps and the flagged weak bullets,
    # not the resume text (Stage 1: the LLM never sees raw documents).
    generated = matching.build_roadmap(role, matches, meta.get("weak_verb_bullets"), weights)

    payload = {
        "role_title": role,
        "role_family": role_family,
        "parse_confidence": {
            "score": meta.get("parse_confidence"),
            "level": meta.get("parse_confidence_level"),
            "notes": meta.get("parse_confidence_notes", []),
        },
        "skills_detected": meta.get("skills_detected", []),
        "requirements": [m.__dict__ for m in matches],
        "roadmap": generated.get("roadmap", []),
        "rewrites": generated.get("rewrites", []),
        "questions": generated.get("questions", []),
        "ats": ats_checks(resume.parse_meta or {}),
    }

    report = Report(
        user_id=user.id,
        resume_id=resume.id,
        jd_id=jd.id,
        match_score=score,
        readiness=readiness,
        headline=head,
        payload=payload,
        engine_version=matching.ENGINE_VERSION,
    )
    db.add(report)
    db.commit()
    audit(db, user.id, "report.created", report.id, score=score, role=role)

    return _to_out(report)


def _to_out(report: Report) -> ReportOut:
    p = report.payload or {}
    return ReportOut(
        id=report.id,
        role_title=p.get("role_title", ""),
        match_score=report.match_score,
        readiness=report.readiness,
        headline=report.headline,
        requirements=p.get("requirements", []),
        roadmap=p.get("roadmap", []),
        rewrites=p.get("rewrites", []),
        questions=p.get("questions", []),
        ats=p.get("ats", []),
        parse_confidence=p.get("parse_confidence"),
        skills_detected=p.get("skills_detected", []),
        engine_version=report.engine_version,
        created_at=report.created_at,
    )


@router.get("/reports", response_model=list[ReportSummary])
def list_reports(
    user: User = Depends(require_consent("resume_processing")),
    db: Session = Depends(get_db),
):
    rows = db.scalars(
        select(Report).where(Report.user_id == user.id).order_by(Report.created_at.desc())
    ).all()
    return [
        ReportSummary(
            id=r.id,
            role_title=(r.payload or {}).get("role_title", ""),
            match_score=r.match_score,
            readiness=r.readiness,
            created_at=r.created_at,
            outcome=r.outcome.status if r.outcome else None,
        )
        for r in rows
    ]


@router.get("/reports/{report_id}", response_model=ReportOut)
def get_report(
    report_id: str,
    user: User = Depends(require_consent("resume_processing")),
    db: Session = Depends(get_db),
):
    report = db.get(Report, report_id)
    if not report or report.user_id != user.id:
        raise HTTPException(404, "No report with that id on your account.")
    return _to_out(report)
