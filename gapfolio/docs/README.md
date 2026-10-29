# GapFolio API

Backend for resume-to-job-description gap analysis: server-side document
parsing, hybrid semantic matching over pgvector, saved reports with progress
tracking, an outcome feedback loop, and enforced consent.

Built against the GapFolio project documentation and pitch deck. This is the
"still to build" list from the prototype, built.

---

## Quick start

```bash
docker compose up --build      # Postgres + pgvector + API on :8000
open http://localhost:8000/docs
```

Without Docker:

```bash
pip install -r requirements.txt
python scripts/migrate.py
uvicorn app.main:app --reload
```

With no configuration it runs on SQLite with offline stubs, so `pytest` and a
local demo work with no keys and no network. Copy `.env.example` to `.env` and
fill in `GAPFOLIO_ANTHROPIC_API_KEY` and `GAPFOLIO_EMBEDDING_API_KEY` for the
real pipeline. `GET /api/health` reports which mode you are in.

---

## What runs when a resume is analysed

```
upload ──▶ parse ──▶ chunk ──▶ embed ──▶ index (pgvector)
                                              │
JD ──▶ extract requirements (LLM) ────────────┤
                                              ▼
                                    hybrid match per requirement
                                              │
                       ┌──────────────────────┼──────────────────────┐
                       ▼                      ▼                      ▼
                 proven/partial/       weighted score          roadmap (LLM)
                    missing          (outcome-calibrated)
                                              │
                                              ▼
                                     report saved ──▶ outcome ──▶ recalibration
```

### Parsing catches what flattened text cannot

`services/parsing.py` runs pdfminer twice with different layout parameters and
compares the resulting reading order. If the order diverges, the resume has
columns — which is the single most common way a resume scrambles inside an ATS,
and it is invisible once the document is a string. DOCX gets the same treatment
for layout tables, text boxes, and contact details parked in the page header.
Scanned PDFs are refused with an explanation rather than silently producing an
empty analysis.

Uploaded bytes are never written to disk.

### Matching is hybrid, and the split matters

Dense embeddings alone get "client engagement" ≈ "customer relationship
management" right, and get Docker vs Kubernetes wrong — those two sit close
together in embedding space precisely because they are both container tooling,
which is the opposite of what a requirement check needs.

So relevance blends dense similarity with IDF-weighted lexical coverage, plus a
small explicit skill ontology (`services/lexical.py`) mapping named
technologies to the categories postings ask for: MySQL implies SQL, Express
implies REST API. The ontology is applied to the **evidence side only** — a
resume saying MySQL satisfies a requirement for SQL, but a resume saying SQL
gets no credit for PostgreSQL tuning it never mentions.

Requirements are scored against **every** chunk of the resume rather than the
ANN top-k. A resume is ~100 chunks, so the full scan is free, and it stops an
index recall miss from turning a real match into a reported gap. A false
"missing" sends a student off to learn something they already have, which is
the error that destroys trust fastest.

### One rule keeps the score honest

A skill in a comma-separated list can never be `proven` — only `partial`. Proof
requires an achievement or a described responsibility. Without this rule the
score rewards keyword stuffing, which is the exact failure the project
documentation criticises in existing tools.

### Thresholds are fitted, not guessed

```bash
python scripts/tune_thresholds.py --verbose
```

Sweeps the blend weight and both decision thresholds against the hand-labeled
pairs in `tests/fixtures/labeled_pairs.json` and prints the configuration that
best reproduces human judgement, plus a confusion matrix.

Current fit with the offline `hash` provider: **87.5% agreement over 24 cases,
4.2% false-missing.** All three disagreements are cases needing real semantics
(React → "user interfaces", login/signup → "authentication"), which is the
expected limit of a stub that does lexical hashing. Re-run with
`--provider voyage` before any pilot and update the constants in
`services/matching.py`.

`tests/test_engine.py` asserts agreement stays above 80% and false-missing
below 10%, so a regression in chunking or the ontology fails a test before it
reaches a student.

### The feedback loop

`POST /api/reports/{id}/outcome` records what happened after applying.
`services/calibration.py` turns those into per-skill weights using
Laplace-smoothed callback rates with a confidence term that shrinks adjustments
on small samples. Skills below 8 samples keep a neutral weight.

`GET /api/research/evaluation` is the RQ2 measurement: callback rate by score
band plus a point-biserial correlation. It refuses to report a verdict below
five outcomes and says plainly when the score is not yet predictive.

Outcome rows are de-identified — no user id — so calibration keeps working
after a student deletes their account, and deletion never destroys the research
dataset. See the policy document for why.

---

## Consent

`require_consent()` is a FastAPI dependency on every route touching resume
data, not a check inside handlers. A route that omits it cannot reach resume
data at all, so the guarantee survives future endpoints written by someone who
has not read the policy.

Three independent scopes: `resume_processing` (required),
`institutional_sharing`, `research_use`. Withdrawal is immediate and
destructive — it deletes stored resumes and vectors in the same request.

Full details, including what a placement officer can and cannot see and the
known gaps: [`docs/DATA_HANDLING_POLICY.md`](DATA_HANDLING_POLICY.md).

---

## API

| Method | Path | Notes |
| --- | --- | --- |
| POST | `/api/auth/register` | Returns a token and the consents still needed |
| POST | `/api/auth/login` | |
| GET | `/api/privacy/policy` | Scopes and processors, for the consent screen |
| POST | `/api/privacy/consents` | Grant or withdraw a scope |
| GET | `/api/privacy/export` | Everything held about the account |
| DELETE | `/api/privacy/account` | Cascading delete |
| POST | `/api/resumes/upload` | PDF, DOCX or TXT — returns ATS warnings |
| POST | `/api/resumes/paste` | Pasted text |
| POST | `/api/analyze` | Full report |
| GET | `/api/reports` · `/api/reports/{id}` | History and detail |
| POST | `/api/reports/{id}/outcome` | The feedback loop |
| GET | `/api/progress` | Score delta, closed gaps, callback rate |
| GET | `/api/institution/dashboard` | Placement cell, consenting students only |
| POST | `/api/research/recalibrate` | Admin; nightly job |
| GET | `/api/research/evaluation` | RQ2 measurement |

Interactive docs at `/docs`.

---

## Tests

```bash
pytest -q      # 18 tests
```

Covers the full journey end to end against a live app, PDF and DOCX parsing
including a scanned-PDF rejection and a table-layout resume, the
listed-vs-proven rule, calibration maths on synthetic outcomes, the consent
gate, and deletion.

---

## Deploying

`docker compose` runs `pgvector/pgvector:pg16` and applies `db/schema.sql`,
which creates the extension and an HNSW cosine index. Set `GAPFOLIO_SECRET_KEY`
to something real, terminate TLS at the proxy, and schedule two jobs:
`POST /api/admin/retention-sweep` and `POST /api/research/recalibrate`, both
nightly, both admin-only.

If you change `GAPFOLIO_EMBEDDING_DIM`, change the dimension in
`db/schema.sql`, reindex every document, and re-run the threshold sweep.

---

## What is deliberately not here

- **No PII redaction before third-party calls.** Resume text goes to the model
  APIs as written. Worth building.
- **No frontend.** The prototype artifact talks to the model API directly; it
  needs rewiring to these endpoints before a pilot.
- **No rate limiting or abuse controls.** Required before public B2C signup.
- **No recruiter-side product.** Phase 4 in the pitch, not started.
