from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_db
from ..models import Institution, User
from ..schemas import LoginIn, RegisterIn, TokenOut
from ..security import (
    audit,
    create_token,
    current_user,
    hash_password,
    has_consent,
    verify_password,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])

REQUIRED_SCOPES = ["resume_processing"]


def _pending(db: Session, user: User) -> list[str]:
    return [s for s in REQUIRED_SCOPES if not has_consent(db, user.id, s)]


@router.post("/register", response_model=TokenOut, status_code=201)
def register(body: RegisterIn, db: Session = Depends(get_db)):
    if db.scalar(select(User).where(User.email == body.email.lower())):
        raise HTTPException(409, "That email is already registered. Sign in instead.")

    # Self-service sign-up creates students only. A placement officer sees other
    # people's consented data, so that role needs an invite code the operator
    # sets; without one configured it cannot be created at all. Nobody can pick
    # their own institution either: it is derived from the verified-by-nothing
    # email domain, which at worst puts a student's own data in front of the
    # officers of the institution they claim, and only if they consent to share.
    if body.role != "student":
        if not settings.officer_invite_code or body.invite_code != settings.officer_invite_code:
            raise HTTPException(403, "Placement-officer accounts need an invitation.")
    domain = (body.institution_domain if body.role != "student" and body.institution_domain
              else body.email.split("@")[-1].lower())
    institution = db.scalar(select(Institution).where(Institution.domain == domain))

    user = User(
        email=body.email.lower(),
        password_hash=hash_password(body.password),
        full_name=body.full_name,
        role=body.role,
        branch=body.branch,
        grad_year=body.grad_year,
        institution_id=institution.id if institution else None,
    )
    db.add(user)
    db.commit()
    audit(db, user.id, "account.created", user.id, role=user.role)
    return TokenOut(
        access_token=create_token(user.id),
        user_id=user.id,
        role=user.role,
        consent_required=_pending(db, user),
    )


@router.post("/login", response_model=TokenOut)
def login(body: LoginIn, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.email == body.email.lower()))
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "That email and password don't match.")
    return TokenOut(
        access_token=create_token(user.id),
        user_id=user.id,
        role=user.role,
        consent_required=_pending(db, user),
    )


@router.get("/me")
def me(user: User = Depends(current_user), db: Session = Depends(get_db)):
    return {
        "id": user.id,
        "email": user.email,
        "full_name": user.full_name,
        "role": user.role,
        "branch": user.branch,
        "grad_year": user.grad_year,
        "institution_id": user.institution_id,
        "consent_required": _pending(db, user),
        "policy_version": settings.consent_version,
    }
