"""Placement-cell views and the research endpoints.

Everything here is aggregate. A placement officer sees a student's score and
gap categories only when that student granted `institutional_sharing`; resume
text is never exposed through this router, whatever the officer's role.
"""
from collections import Counter

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Report, User
from ..security import audit, current_user, has_consent, require_officer
from ..services import calibration

router = APIRouter(prefix="/api/institution", tags=["institution"])

BANDS = [(0, 40), (40, 55), (55, 70), (70, 85), (85, 101)]


def _shared_students(db: Session, institution_id: str) -> list[User]:
    students = db.scalars(
        select(User).where(
            User.institution_id == institution_id, User.role == "student"
        )
    ).all()
    return [s for s in students if has_consent(db, s.id, "institutional_sharing")]


@router.get("/dashboard")
def dashboard(
    officer: User = Depends(require_officer), db: Session = Depends(get_db)
):
    if not officer.institution_id:
        raise HTTPException(400, "This account isn't linked to an institution yet.")

    students = _shared_students(db, officer.institution_id)
    rows, gap_counter = [], Counter()

    for s in students:
        latest = db.scalars(
            select(Report)
            .where(Report.user_id == s.id)
            .order_by(Report.created_at.desc())
            .limit(1)
        ).first()
        if not latest:
            continue
        reqs = (latest.payload or {}).get("requirements", [])
        gaps = [r["label"] for r in reqs if r.get("status") != "proven"]
        gap_counter.update(g.lower() for g in gaps)
        rows.append(
            {
                "student_id": s.id,
                "name": s.full_name or s.email.split("@")[0],
                "branch": s.branch,
                "score": latest.match_score,
                "readiness": latest.readiness,
                "top_gap": gaps[0] if gaps else "—",
                "applications": db.scalar(
                    select(func.count(Report.id)).where(Report.user_id == s.id)
                ),
                "updated_at": latest.created_at,
            }
        )

    scores = [r["score"] for r in rows]
    distribution = [
        {"band": f"{lo}-{hi - 1}", "n": sum(1 for s in scores if lo <= s < hi)}
        for lo, hi in BANDS
    ]
    audit(db, officer.id, "dashboard.viewed", officer.institution_id, students=len(rows))

    return {
        "students_shared": len(students),
        "students_with_reports": len(rows),
        "median_score": round(sorted(scores)[len(scores) // 2], 1) if scores else None,
        "at_risk": sum(1 for s in scores if s < 50),
        "distribution": distribution,
        "common_gaps": [{"skill": k, "students": v} for k, v in gap_counter.most_common(8)],
        "students": sorted(rows, key=lambda r: r["score"]),
    }


@router.get("/student/{student_id}")
def student_detail(
    student_id: str, officer: User = Depends(require_officer), db: Session = Depends(get_db)
):
    student = db.get(User, student_id)
    if not student or student.institution_id != officer.institution_id:
        raise HTTPException(404, "No such student at your institution.")
    if not has_consent(db, student.id, "institutional_sharing"):
        raise HTTPException(403, "This student hasn't shared their results with the placement cell.")

    reports = db.scalars(
        select(Report).where(Report.user_id == student.id).order_by(Report.created_at.desc())
    ).all()
    audit(db, officer.id, "student.viewed", student_id)
    return {
        "name": student.full_name,
        "branch": student.branch,
        # Gap labels and scores only — never resume text.
        "history": [
            {
                "role": (r.payload or {}).get("role_title", ""),
                "score": r.match_score,
                "readiness": r.readiness,
                "gaps": [q["label"] for q in (r.payload or {}).get("requirements", []) if q.get("status") != "proven"],
                "at": r.created_at,
            }
            for r in reports
        ],
    }


# --- research / calibration ----------------------------------------------

research = APIRouter(prefix="/api/research", tags=["research"])


@research.post("/recalibrate")
def recalibrate(
    role_family: str = "general",
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    if user.role != "admin":
        raise HTTPException(403, "Calibration runs are admin-only.")
    result = calibration.recalibrate(db, role_family)
    audit(db, user.id, "model.recalibrated", role_family, **result)
    return result


@research.get("/evaluation")
def evaluation(user: User = Depends(current_user), db: Session = Depends(get_db)):
    """RQ2 measurement: does the score predict callbacks? Aggregate only."""
    return calibration.evaluate(db)
