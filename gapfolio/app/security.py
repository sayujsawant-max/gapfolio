"""Auth and the consent gate.

`require_consent` is deliberately a hard dependency rather than a check inside
each handler: no code path can process a resume unless a matching consent row
exists. That is the technical half of the data-handling policy in
docs/DATA_HANDLING_POLICY.md.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
import base64

import bcrypt
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import settings
from .db import get_db
from .models import AuditLog, Consent, User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

ALGORITHM = "HS256"


def _prepare(raw: str) -> bytes:
    """bcrypt silently truncates at 72 bytes, which would make two long
    passwords sharing a prefix interchangeable. SHA-256 first, base64 second,
    so every password reaches bcrypt as a fixed 44-byte value."""
    digest = hashlib.sha256(raw.encode("utf-8")).digest()
    return base64.b64encode(digest)


def hash_password(raw: str) -> str:
    return bcrypt.hashpw(_prepare(raw), bcrypt.gensalt(rounds=12)).decode()


def verify_password(raw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(_prepare(raw), hashed.encode())
    except ValueError:
        return False


def create_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_minutes)
    return jwt.encode({"sub": user_id, "exp": expire}, settings.secret_key, algorithm=ALGORITHM)


def hash_ip(request: Request) -> str:
    """Consent records need proof of origin without storing an IP address."""
    ip = request.client.host if request.client else ""
    return hashlib.sha256(f"{ip}{settings.secret_key}".encode()).hexdigest()[:32]


def current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
) -> User:
    creds_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Sign in again — your session expired.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
    except JWTError:
        raise creds_error
    user = db.get(User, user_id) if user_id else None
    if user is None or not user.is_active:
        raise creds_error
    return user


def require_officer(user: User = Depends(current_user)) -> User:
    if user.role not in ("placement_officer", "admin"):
        raise HTTPException(403, "This view is for placement staff.")
    return user


def has_consent(db: Session, user_id: str, scope: str) -> bool:
    """Latest row wins, so a withdrawal immediately revokes access."""
    row = db.scalars(
        select(Consent)
        .where(Consent.user_id == user_id, Consent.scope == scope)
        .order_by(Consent.created_at.desc())
        .limit(1)
    ).first()
    return bool(row and row.granted)


def require_consent(scope: str = "resume_processing"):
    def dependency(
        user: User = Depends(current_user), db: Session = Depends(get_db)
    ) -> User:
        if not has_consent(db, user.id, scope):
            raise HTTPException(
                status_code=451,
                detail={
                    "error": "consent_required",
                    "scope": scope,
                    "policy_version": settings.consent_version,
                    "message": "Agree to how your resume is handled before it can be processed.",
                },
            )
        return user

    return dependency


def audit(db: Session, actor_id: str, action: str, target: str = "", **detail) -> None:
    db.add(AuditLog(actor_id=actor_id, action=action, target=target, detail=detail))
    db.commit()
