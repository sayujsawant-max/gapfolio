"""Outcome capture and progress across applications.

The outcome row is written with a snapshot of the requirement statuses at the
time of applying, not a pointer to the live report. Reports can be deleted and
resumes get purged on a retention schedule; the snapshot is what makes the
feedback loop durable.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Outcome, Report, User
from ..schemas import OutcomeIn, ProgressOut
from ..security import audit, current_user, require_consent
from ..services.calibration import CALLBACK

router = APIRouter(prefix="/api", tags=["progress"])


@router.post("/reports/{report_id}/outcome", status_code=201)
def record_outcome(
    report_id: str,
    body: OutcomeIn,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    report = db.get(Report, report_id)
    if not report or report.user_id != user.id:
        raise HTTPException(404, "No report with that id on your account.")

    snapshot = [
        {"label": r.get("label"), "status": r.get("status"), "weight": r.get("weight")}
        for r in (report.payload or {}).get("requirements", [])
    ]

    existing = db.scalar(select(Outcome).where(Outcome.report_id == report_id))
    if existing:
        existing.status = body.status
        existing.days_to_response = body.days_to_response
        existing.note = body.note
    else:
        db.add(
            Outcome(
                report_id=report_id,
                status=body.status,
                score_at_apply=report.match_score,
                requirement_snapshot=snapshot,
                days_to_response=body.days_to_response,
                note=body.note,
            )
        )
    db.commit()
    audit(db, user.id, "outcome.recorded", report_id, status=body.status)
    return {"recorded": True, "status": body.status}


@router.get("/progress", response_model=ProgressOut)
def progress(
    user: User = Depends(require_consent("resume_processing")),
    db: Session = Depends(get_db),
):
    reports = db.scalars(
        select(Report).where(Report.user_id == user.id).order_by(Report.created_at.asc())
    ).all()
    if not reports:
        return ProgressOut(
            reports=0, first_score=None, latest_score=None, delta=None, callbacks=0,
            applications=0, callback_rate=None, closed_gaps=[], open_gaps=[], timeline=[],
        )

    first, latest = reports[0], reports[-1]

    def statuses(r: Report) -> dict[str, str]:
        return {
            (q.get("label") or "").lower(): q.get("status")
            for q in (r.payload or {}).get("requirements", [])
        }

    # A gap counts as closed when a skill that was missing or unproven in an
    # earlier report comes back proven in a later one.
    ever_gap: dict[str, str] = {}
    for r in reports:
        for label, st in statuses(r).items():
            if st != "proven" and label not in ever_gap:
                ever_gap[label] = st
    latest_st = statuses(latest)
    closed = sorted(l for l, _ in ever_gap.items() if latest_st.get(l) == "proven")
    open_gaps = sorted(l for l, st in latest_st.items() if st != "proven")

    outcomes = [r.outcome for r in reports if r.outcome]
    callbacks = sum(1 for o in outcomes if o.status in CALLBACK)

    timeline = [
        {
            "report_id": r.id,
            "role": (r.payload or {}).get("role_title", ""),
            "score": r.match_score,
            "readiness": r.readiness,
            "outcome": r.outcome.status if r.outcome else None,
            "at": r.created_at,
        }
        for r in reports
    ]

    return ProgressOut(
        reports=len(reports),
        first_score=first.match_score,
        latest_score=latest.match_score,
        delta=round(latest.match_score - first.match_score, 1),
        callbacks=callbacks,
        applications=len(outcomes),
        callback_rate=round(callbacks / len(outcomes), 2) if outcomes else None,
        closed_gaps=closed[:20],
        open_gaps=open_gaps[:20],
        timeline=timeline,
    )
