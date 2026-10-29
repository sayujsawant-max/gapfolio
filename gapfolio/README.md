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

Embeddings come from a local Sentence-Transformers model
(`all-MiniLM-L6-v2`, 384-d): no API key, no per-call cost. Install CPU PyTorch
first (`pip install torch --index-url https://download.pytorch.org/whl/cpu`);
the model downloads on first use (~90 MB). With no other configuration it runs
on SQLite and uses rule-based text for the roadmap, so a local demo needs no
keys. Set `GAPFOLIO_ANTHROPIC_API_KEY` to have Claude write the roadmap text;
scoring never depends on it. `GET /api/health` reports which mode you are in,
and `pytest` runs on the offline `hash` stub so it needs no model.

The built React app (`../gapfolio-ui`) is served from the same process, so
`http://localhost:8000` is the whole product. For one-origin deployment see the
`Dockerfile` and `render.yaml` in the parent directory.

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

`services/parsing.py` reads the PDF's text-line geometry. Two groups of
substantial lines with different left edges that sit side by side and share
vertical extent are two columns — the most common way a resume scrambles inside
an ATS, and invisible once the document is a string. (The first version compared
two text extractions and detected nothing, because both interleave a two-column
page identically; geometry is what works.) Short right-aligned dates and
indented bullets are tolerated. DOCX gets the same treatment for layout tables,
text boxes, and contact details parked in the page header. Fonts that are named
but not embedded are flagged. Scanned PDFs are refused with an explanation
rather than silently producing an empty analysis.

Every parse also gets a **parse confidence** score (text density, recognised
sections, contact details, layout defects, bullet count). Below 0.6 the UI
warns that the analysis is provisional. `services/entities.py` segments the
resume into sections — a synonym rulebook first, then an embedding-plus-cues
classifier for headings like "Things I Built" — and extracts skills with a
phrase ruler. ATS checks carry a group: `parse` (can a machine read it?) or
`readability` (does it read well?), kept separate on purpose.

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

Sweeps the blend weight and both decision thresholds against the 76 hand-labeled
pairs in `tests/fixtures/labeled_pairs.json` (7 resumes) and reports three
numbers, because one would flatter the fit: agreement on the development cases,
agreement on **held-out** cases written afterwards, and leave-one-resume-out
cross-validation. Thresholds are per embedding provider, because cosine
similarity has a different scale for every model (`_FITTED` in
`services/matching.py`).

With SBERT: **90.4% agreement on the 52 held-out cases** (1.9% false-missing)
when fitted on the 24 development cases only; **90.8%** leave-one-resume-out;
92.1% in-sample when fitted on all 76. The offline stub scores 80.8% held-out
with 9.6% false-missing, which is why it is for tests only. The labels come from
a single annotator.

`python scripts/evaluate.py` measures the rest (section extraction, skill F1,
ATS defect injection, roadmap grounding, latency). `tests/test_stage3.py`
guards the shipped configuration: with the real model, agreement must stay
above 88% and false-missing below 3%, so a regression in chunking or the
ontology fails a test before it reaches a student.

### The roadmap is ranked by arithmetic, and the LLM only explains it

Gaps are ranked by `impact / sqrt(effort_days)`, where impact is the score a
closed gap would add and effort comes from the gap type (rewrite a bullet: 1
day; build a project behind a listed skill: 4; learn a tool: 10; a whole
concept: 14). The LLM never sees the resume — only the ranked gaps, the
evidence for each, and the flagged weak bullets — and whatever it returns is
validated: a roadmap item must name a real gap, a rewrite must quote a flagged
bullet verbatim and may not introduce a number. With no API key the whole
roadmap is templated and equally grounded.

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
pytest -q      # 43 tests
```

Covers the full journey end to end against a live app, PDF and DOCX parsing
including a scanned-PDF rejection and a table-layout resume, column detection
(including the right-aligned-dates false positive), the tokeniser, modal-language
tagging, gap ranking, roadmap grounding, section segmentation, the
listed-vs-proven rule, calibration maths on synthetic outcomes, the consent
gate, and deletion.

---

## Deploying

`docker compose` runs `pgvector/pgvector:pg16` and applies `db/schema.sql`,
which creates the extension and an HNSW cosine index. Set `GAPFOLIO_SECRET_KEY`
to something real, terminate TLS at the proxy, and schedule two jobs:
`POST /api/admin/retention-sweep` and `POST /api/research/recalibrate`, both
nightly, both admin-only.

If you change the embedding model or `GAPFOLIO_EMBEDDING_DIM`, reindex every
document (a resume indexed under a different dimension is re-indexed on its next
analysis) and re-run the threshold sweep for that provider.

Public hardening that is in: self-service sign-up creates students only
(placement officers need `GAPFOLIO_OFFICER_INVITE_CODE`), a per-client rate limit
(`GAPFOLIO_RATE_LIMIT_*`, in-memory, per process), a 5 MB upload cap, and a
random `GAPFOLIO_SECRET_KEY`. The `Dockerfile` and `render.yaml` in the parent
directory build the single-origin image; neither has been run on a Docker host
yet.

---

## What is deliberately not here

- **No PII redaction before third-party calls.** Resume text goes to the model
  APIs as written when an Anthropic key is set. Worth building.
- **No second annotator.** The 76 labelled cases are from one annotator, so
  inter-annotator agreement is not reported.
- **No human rating of roadmap quality.** Grounding is checked automatically; how
  useful the advice is has not been rated by people.
- **No recruiter-side product.** Phase 4 in the pitch, not started.
- **spaCy is not used.** The skill ruler and section classifier are
  self-contained; spaCy could not be loaded under this machine's application
  control policy.
