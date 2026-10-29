"""End-to-end flow over the real app, on a throwaway SQLite database."""
import io
import os
import tempfile

import pytest

os.environ["GAPFOLIO_DATABASE_URL"] = f"sqlite:///{tempfile.mkdtemp()}/test.db"
os.environ["GAPFOLIO_EMBEDDING_PROVIDER"] = "hash"
os.environ["GAPFOLIO_EMBEDDING_DIM"] = "1024"  # the offline stub was tuned at 1024-d
os.environ["GAPFOLIO_ANTHROPIC_API_KEY"] = ""  # forces heuristic fallbacks

from fastapi.testclient import TestClient  # noqa: E402

from app.db import init_db  # noqa: E402
from app.main import app  # noqa: E402

RESUME = """ANANYA IYER
B.Tech Computer Science, 2027 | Pune | ananya.iyer@example.com | github.com/ananya

EDUCATION
Vishwakarma Institute of Technology, Pune - B.Tech CSE, CGPA 8.4/10

PROJECTS
Campus Food Ordering App
- Built a REST API in Express serving 120 students with JWT authentication
- Wrote 40 unit tests taking coverage to 82%

Movie Recommendation System
- Helped the team with the data cleaning tasks

INTERNSHIP
Web Development Intern, Nexlify Solutions (May-July 2026)
- Fixed 14 bugs reported through code review by the senior developer

SKILLS
HTML, CSS, JavaScript, React, Python, Docker, MySQL, Git
"""

JD = """Software Engineer I - Backend (Fresher)
Bengaluru | Full-time

What you'll do
- Design and build REST APIs in Node.js or Python that stay reliable under load
- Write and optimise SQL queries against PostgreSQL including query tuning
- Containerise services with Docker and ship them through CI/CD pipelines
- Write unit and integration tests with meaningful coverage on every pull request
- Work with AWS services such as EC2, S3 and Lambda

What we're looking for
- Hands-on experience with Git and code review workflows
- Strong fundamentals in data structures and algorithms
"""


@pytest.fixture(scope="module")
def client():
    init_db()
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def student(client):
    r = client.post(
        "/api/auth/register",
        json={
            "email": "ananya@vit.example.edu",
            "password": "correct-horse-battery",
            "full_name": "Ananya Iyer",
            "branch": "CSE",
        },
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert "resume_processing" in body["consent_required"]
    return {"Authorization": f"Bearer {body['access_token']}"}


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["llm"] == "fallback heuristics"


def test_resume_blocked_before_consent(client, student):
    r = client.post("/api/resumes/paste", json={"text": RESUME}, headers=student)
    assert r.status_code == 451
    assert r.json()["detail"]["error"] == "consent_required"


def test_consent_then_upload_and_analyze(client, student):
    r = client.post(
        "/api/privacy/consents",
        json={"scope": "resume_processing", "granted": True},
        headers=student,
    )
    assert r.status_code == 201

    r = client.post("/api/resumes/paste", json={"text": RESUME}, headers=student)
    assert r.status_code == 201, r.text
    resume_id = r.json()["id"]
    assert r.json()["purge_after"] is not None

    r = client.post(
        "/api/analyze",
        json={"resume_id": resume_id, "jd_text": JD},
        headers=student,
    )
    assert r.status_code == 201, r.text
    report = r.json()
    assert 0 <= report["match_score"] <= 100
    assert report["requirements"], "requirement extraction produced nothing"
    statuses = {q["status"] for q in report["requirements"]}
    assert statuses <= {"proven", "partial", "missing"}
    assert report["roadmap"], "no roadmap generated"
    assert any(c["status"] != "pass" for c in report["ats"])
    pytest.report_id = report["id"]


def test_docx_upload_parses(client, student):
    docx = pytest.importorskip("docx")
    doc = docx.Document()
    for line in RESUME.splitlines():
        doc.add_paragraph(line)
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)

    r = client.post(
        "/api/resumes/upload",
        files={"file": ("resume.docx", buf.read(),
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        headers=student,
    )
    assert r.status_code == 201, r.text
    assert r.json()["source"] == "docx"
    assert r.json()["char_count"] > 400


def test_rejects_unsupported_format(client, student):
    r = client.post(
        "/api/resumes/upload",
        files={"file": ("resume.pages", b"x" * 500, "application/octet-stream")},
        headers=student,
    )
    assert r.status_code == 422


def test_outcome_and_progress(client, student):
    r = client.post(
        f"/api/reports/{pytest.report_id}/outcome",
        json={"status": "interview", "days_to_response": 6},
        headers=student,
    )
    assert r.status_code == 201

    p = client.get("/api/progress", headers=student).json()
    assert p["reports"] >= 1
    assert p["applications"] == 1
    assert p["callbacks"] == 1
    assert p["callback_rate"] == 1.0
    assert p["timeline"][0]["outcome"] == "interview"


def test_export_and_delete(client):
    r = client.post(
        "/api/auth/register",
        json={"email": "temp@vit.example.edu", "password": "another-long-password"},
    )
    headers = {"Authorization": f"Bearer {r.json()['access_token']}"}
    client.post(
        "/api/privacy/consents",
        json={"scope": "resume_processing", "granted": True},
        headers=headers,
    )
    client.post("/api/resumes/paste", json={"text": RESUME}, headers=headers)

    export = client.get("/api/privacy/export", headers=headers).json()
    assert export["resumes"][0]["text"].startswith("ANANYA")

    d = client.delete("/api/privacy/account", headers=headers)
    assert d.status_code == 200
    assert d.json()["deleted"] is True
    # Token is now dead.
    assert client.get("/api/auth/me", headers=headers).status_code == 401


def test_withdrawing_consent_purges_resumes(client):
    r = client.post(
        "/api/auth/register",
        json={"email": "withdraw@vit.example.edu", "password": "yet-another-password"},
    )
    headers = {"Authorization": f"Bearer {r.json()['access_token']}"}
    client.post("/api/privacy/consents", json={"scope": "resume_processing", "granted": True}, headers=headers)
    client.post("/api/resumes/paste", json={"text": RESUME}, headers=headers)
    assert len(client.get("/api/resumes", headers=headers).json()) == 1

    client.post("/api/privacy/consents", json={"scope": "resume_processing", "granted": False}, headers=headers)
    # Consent is gone, so the listing is now gated too.
    assert client.get("/api/resumes", headers=headers).status_code == 451

    client.post("/api/privacy/consents", json={"scope": "resume_processing", "granted": True}, headers=headers)
    assert client.get("/api/resumes", headers=headers).json() == []
