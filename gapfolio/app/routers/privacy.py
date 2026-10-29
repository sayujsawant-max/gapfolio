"""Consent, export and deletion.

The three rights a student pilot has to support in practice: know what is held,
take it away, and have it erased. Deletion removes every identifiable record
and the resume text; it deliberately leaves the de-identified `outcomes` row,
which carries no user id and cannot be traced back — that is what keeps the
research dataset intact without holding anyone's data against their wishes.
"""
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Request
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_db
from ..models import Consent, Embedding, Report, Resume, User
from ..schemas import ConsentIn, ConsentOut
from ..security import audit, current_user, hash_ip

router = APIRouter(prefix="/api/privacy", tags=["privacy"])

SCOPE_TEXT = {
    "resume_processing": (
        "Your resume text is sent to our analysis models and stored so you can see your reports. "
        "It is never shared with recruiters or employers."
    ),
    "institutional_sharing": (
        "Your match scores and gap categories are visible to your college's placement cell. "
        "Your resume text and personal contact details are not."
    ),
    "research_use": (
        "Anonymised gap patterns and application outcomes may be used in published research. "
        "Nothing that identifies you is included."
    ),
}


@router.get("/policy")
def policy():
    return {
        "version": settings.consent_version,
        "retention_days": settings.retention_days,
        "scopes": [
            {"scope": k, "description": v, "required": k == "resume_processing"}
            for k, v in SCOPE_TEXT.items()
        ],
        "processors": [
            {"name": "Anthropic", "purpose": "requirement extraction and roadmap generation", "retains": "no training use via API"},
            {"name": "Embedding provider", "purpose": "semantic matching", "retains": "no"},
        ],
    }


@router.get("/consents", response_model=list[ConsentOut])
def list_consents(user: User = Depends(current_user), db: Session = Depends(get_db)):
    rows = db.scalars(
        select(Consent).where(Consent.user_id == user.id).order_by(Consent.created_at.desc())
    ).all()
    latest: dict[str, Consent] = {}
    for row in rows:
        latest.setdefault(row.scope, row)
    return list(latest.values())


@router.post("/consents", response_model=ConsentOut, status_code=201)
def set_consent(
    body: ConsentIn,
    request: Request,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    row = Consent(
        user_id=user.id,
        scope=body.scope,
        granted=body.granted,
        policy_version=settings.consent_version,
        ip_hash=hash_ip(request),
    )
    db.add(row)
    db.commit()
    audit(db, user.id, "consent.granted" if body.granted else "consent.withdrawn", body.scope)

    # Withdrawing resume processing purges the material it covered.
    if body.scope == "resume_processing" and not body.granted:
        _purge_resumes(db, user.id)
        audit(db, user.id, "data.purged", "resumes", reason="consent_withdrawn")
    return row


@router.get("/export")
def export(user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Everything held about this account, in one JSON payload."""
    resumes = db.scalars(select(Resume).where(Resume.user_id == user.id)).all()
    reports = db.scalars(select(Report).where(Report.user_id == user.id)).all()
    consents = db.scalars(select(Consent).where(Consent.user_id == user.id)).all()
    audit(db, user.id, "data.exported", user.id)
    return {
        "account": {
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
            "branch": user.branch,
            "grad_year": user.grad_year,
            "created_at": user.created_at,
        },
        "consents": [
            {"scope": c.scope, "granted": c.granted, "version": c.policy_version, "at": c.created_at}
            for c in consents
        ],
        "resumes": [
            {"id": r.id, "filename": r.filename, "source": r.source, "text": r.text, "at": r.created_at}
            for r in resumes
        ],
        "reports": [
            {
                "id": r.id,
                "score": r.match_score,
                "readiness": r.readiness,
                "payload": r.payload,
                "at": r.created_at,
            }
            for r in reports
        ],
    }


@router.delete("/account", status_code=200)
def delete_account(user: User = Depends(current_user), db: Session = Depends(get_db)):
    _purge_resumes(db, user.id)
    report_ids = db.scalars(select(Report.id).where(Report.user_id == user.id)).all()
    # Outcomes are de-identified and survive; detach them from the deleted reports.
    from ..models import Outcome

    for oid in report_ids:
        o = db.scalar(select(Outcome).where(Outcome.report_id == oid))
        if o is not None:
            o.report_id = f"deleted:{oid[:8]}"
    db.commit()

    audit(db, user.id, "account.deleted", user.id, reports=len(report_ids))
    db.execute(delete(Report).where(Report.user_id == user.id))
    db.execute(delete(Consent).where(Consent.user_id == user.id))
    db.delete(db.get(User, user.id))
    db.commit()
    return {
        "deleted": True,
        "reports_removed": len(report_ids),
        "note": "De-identified outcome records are retained for research and cannot be linked to you.",
    }


def _purge_resumes(db: Session, user_id: str) -> int:
    resumes = db.scalars(select(Resume).where(Resume.user_id == user_id)).all()
    for r in resumes:
        db.execute(
            delete(Embedding).where(
                Embedding.owner_type == "resume", Embedding.owner_id == r.id
            )
        )
        db.delete(r)
    db.commit()
    return len(resumes)


def run_retention_sweep(db: Session) -> dict:
    """Delete resume text past its retention date. Call from a nightly job."""
    now = datetime.now(timezone.utc)
    stale = db.scalars(select(Resume).where(Resume.purge_after.isnot(None))).all()
    removed = 0
    for r in stale:
        deadline = r.purge_after
        if deadline and deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
        if deadline and deadline <= now:
            db.execute(
                delete(Embedding).where(
                    Embedding.owner_type == "resume", Embedding.owner_id == r.id
                )
            )
            db.delete(r)
            removed += 1
    db.commit()
    return {"resumes_purged": removed, "checked": len(stale), "at": now}


def default_purge_date() -> datetime:
    return datetime.now(timezone.utc) + timedelta(days=settings.retention_days)
