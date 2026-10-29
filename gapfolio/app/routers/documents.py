from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_db
from ..models import Resume, User
from ..schemas import PasteIn, ResumeOut
from ..security import audit, require_consent
from ..services import vectorstore
from ..services.parsing import ParseError, ats_checks, parse_document, parse_text
from .privacy import default_purge_date

router = APIRouter(prefix="/api/resumes", tags=["resumes"])


def _store(db: Session, user: User, parsed, filename: str) -> Resume:
    resume = Resume(
        user_id=user.id,
        filename=filename,
        source=parsed.source,
        text=parsed.text if settings.store_raw_text else "",
        parse_meta=parsed.meta,
        purge_after=default_purge_date(),
    )
    db.add(resume)
    db.commit()
    vectorstore.index_document(db, "resume", resume.id, parsed.text)
    audit(db, user.id, "resume.uploaded", resume.id, source=parsed.source)
    return resume


def _out(resume: Resume, parsed) -> ResumeOut:
    return ResumeOut(
        id=resume.id,
        filename=resume.filename,
        source=resume.source,
        char_count=len(parsed.text),
        warnings=[c for c in ats_checks(parsed.meta) if c["status"] != "pass"],
        parse_confidence=parsed.meta.get("parse_confidence"),
        parse_confidence_level=parsed.meta.get("parse_confidence_level"),
        parse_confidence_notes=parsed.meta.get("parse_confidence_notes", []),
        skills_detected=parsed.meta.get("skills_detected", []),
        purge_after=resume.purge_after,
    )


@router.post("/upload", response_model=ResumeOut, status_code=201)
async def upload(
    file: UploadFile = File(...),
    user: User = Depends(require_consent("resume_processing")),
    db: Session = Depends(get_db),
):
    data = await file.read()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, "That file is over 5 MB. Export a smaller PDF and try again.")
    try:
        parsed = parse_document(data, file.filename or "")
    except ParseError as e:
        raise HTTPException(422, str(e))

    resume = _store(db, user, parsed, file.filename or "resume")
    return _out(resume, parsed)


@router.post("/paste", response_model=ResumeOut, status_code=201)
def paste(
    body: PasteIn,
    user: User = Depends(require_consent("resume_processing")),
    db: Session = Depends(get_db),
):
    parsed = parse_text(body.text)
    resume = _store(db, user, parsed, body.filename)
    return _out(resume, parsed)


@router.get("", response_model=list[ResumeOut])
def list_resumes(
    user: User = Depends(require_consent("resume_processing")),
    db: Session = Depends(get_db),
):
    rows = db.scalars(
        select(Resume).where(Resume.user_id == user.id).order_by(Resume.created_at.desc())
    ).all()
    return [
        ResumeOut(
            id=r.id,
            filename=r.filename,
            source=r.source,
            char_count=len(r.text or ""),
            warnings=[],
            purge_after=r.purge_after,
        )
        for r in rows
    ]


@router.delete("/{resume_id}", status_code=200)
def delete_resume(
    resume_id: str,
    user: User = Depends(require_consent("resume_processing")),
    db: Session = Depends(get_db),
):
    resume = db.get(Resume, resume_id)
    if not resume or resume.user_id != user.id:
        raise HTTPException(404, "No resume with that id on your account.")
    vectorstore.drop_document(db, "resume", resume.id)
    db.delete(resume)
    db.commit()
    audit(db, user.id, "resume.deleted", resume_id)
    return {"deleted": True}
