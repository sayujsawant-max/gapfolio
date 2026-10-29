import pathlib
import threading
import time
from collections import defaultdict, deque

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from .config import settings, is_postgres
from .db import get_db, init_db
from .routers import analysis, auth, documents, institution, privacy, progress
from .routers.privacy import run_retention_sweep
from .security import current_user
from .models import User
from .services.embeddings import effective_provider

app = FastAPI(
    title=settings.app_name,
    version="3.0.0",
    description="Resume-to-JD gap analysis. Semantic matching, saved reports, outcome feedback.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- rate limiting ----------------------------------------------------------
# In-memory sliding window per client. Enough to stop one visitor from running
# up the analysis cost or hammering sign-in; it resets on restart and is per
# process, so a multi-worker deployment should move it to the proxy or Redis.
# The client is taken from the proxy's header when present, because behind a
# tunnel or load balancer every request otherwise appears to come from one IP.
_HEAVY = ("/api/analyze", "/api/resumes/upload", "/api/resumes/paste", "/api/auth/")
_hits: dict[tuple[str, bool], deque] = defaultdict(deque)
_lock = threading.Lock()


def _client(request: Request) -> str:
    fwd = request.headers.get("cf-connecting-ip") or request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() or (request.client.host if request.client else "?")


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    path = request.url.path
    if not path.startswith("/api/") or path == "/api/health" or request.method == "OPTIONS":
        return await call_next(request)
    heavy = path.startswith(_HEAVY) and request.method == "POST"
    limit = settings.rate_limit_heavy if heavy else settings.rate_limit_general
    now = time.monotonic()
    with _lock:
        window = _hits[(_client(request), heavy)]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= limit:
            retry = int(60 - (now - window[0])) + 1
            return JSONResponse(
                {"detail": f"Too many requests. Try again in {retry} seconds."},
                status_code=429, headers={"Retry-After": str(retry)},
            )
        window.append(now)
    return await call_next(request)

app.include_router(auth.router)
app.include_router(privacy.router)
app.include_router(documents.router)
app.include_router(analysis.router)
app.include_router(progress.router)
app.include_router(institution.router)
app.include_router(institution.research)


@app.on_event("startup")
def startup() -> None:
    init_db()
    # Load the embedding model in the background so the first analysis does not
    # pay for it; /api/health reports "loading" until it is ready.
    threading.Thread(target=effective_provider, daemon=True).start()


@app.get("/api/health")
def health():
    from .services import embeddings

    ready = embeddings._sbert is not None or embeddings._sbert_failed or settings.embedding_provider != "sbert"
    return {
        "status": "ok",
        "database": "postgres+pgvector" if is_postgres() else "sqlite (dev)",
        "embeddings": effective_provider() if ready else "sbert (loading)",
        "llm": "configured" if settings.anthropic_api_key else "fallback heuristics",
        "policy_version": settings.consent_version,
    }


def _mount_ui() -> None:
    """Serve the built React app from the same origin as the API, so one public
    URL (and one tunnel or one host) carries the whole product and the browser
    needs no CORS."""
    dist = pathlib.Path(settings.ui_dist)
    if not dist.is_absolute():
        dist = (pathlib.Path(__file__).resolve().parents[1] / dist).resolve()
    if settings.serve_ui and (dist / "index.html").exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="ui")


@app.post("/api/admin/retention-sweep")
def retention_sweep(user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Nightly job endpoint. Deletes resume text past its retention date."""
    if user.role != "admin":
        from fastapi import HTTPException

        raise HTTPException(403, "Admin only.")
    return run_retention_sweep(db)


_mount_ui()
