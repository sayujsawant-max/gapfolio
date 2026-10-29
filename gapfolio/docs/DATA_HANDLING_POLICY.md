# Data Handling Policy

**Version 2026-08-v1** · Applies to the GapFolio pilot at partner colleges.

This document exists because a resume is not neutral data. It carries a
student's name, phone number, address, caste-indicative surname, college, and
employment history in a single file. A placement cell asking students to upload
one is asking for real trust, and the project documentation lists data privacy
as the first open risk. This is the answer to it.

The commitments below are enforced in code, not just stated here. Where a rule
is enforced, the enforcing module is named.

---

## 1. What is collected

| Data | Why | Where it lives |
| --- | --- | --- |
| Email, name, branch, graduating year | Account identity and batch grouping | `users` |
| Resume text (parsed from the upload) | The analysis itself | `resumes.text` |
| Structural signals (columns, tables, bullet counts) | ATS risk checks | `resumes.parse_meta` |
| Job description text | Matching; deduplicated across all users | `job_descriptions` |
| Vector chunks of resume and JD | Semantic matching | `embeddings` |
| Reports (score, gaps, roadmap) | Progress over time | `reports` |
| Application outcomes | Model calibration and research | `outcomes` |
| Consent records | Proof of what was agreed and when | `consents` |
| Staff access records | Accountability for who viewed what | `audit_log` |

**Not collected:** date of birth, gender, caste, religion, category, family
income, disability status, photographs, government identifiers. The parser does
not extract them and no endpoint accepts them. If a student's resume contains
such details in its text, they are stored only as part of the resume text and
are never surfaced to a placement officer.

**Original files are never stored.** Uploads are parsed in memory and the bytes
are discarded (`services/parsing.py`). Only extracted text is retained.

---

## 2. Consent

Three separate scopes. Granting one does not grant another.

| Scope | Covers | Required? |
| --- | --- | --- |
| `resume_processing` | Parsing, storing and analysing your resume | Yes — nothing works without it |
| `institutional_sharing` | Your scores and gap categories visible to your placement cell | No |
| `research_use` | Anonymised patterns used in published research | No |

Enforced by `require_consent()` in `app/security.py`, applied as a FastAPI
dependency on every resume and analysis route. It is a dependency rather than
an in-handler check specifically so that no future endpoint can forget it — a
route without the dependency cannot reach resume data at all.

Consent is an **append-only ledger**. Granting and withdrawing are both rows,
each stamped with the policy version and a salted hash of the originating IP
(never the IP itself). There is always a provable history of what was agreed to
and when.

**Withdrawal is immediate and destructive.** Withdrawing `resume_processing`
deletes every stored resume and its vector chunks in the same request
(`routers/privacy.py`). It is not a flag; the data goes.

**Re-consent on policy change.** Bumping `GAPFOLIO_CONSENT_VERSION` causes the
stored version to mismatch, and students are asked again. Silent expansion of
scope under an old agreement is not possible.

---

## 3. What the placement cell can and cannot see

A placement officer sees a student **only** if that student granted
`institutional_sharing`. Non-consenting students are absent from the dashboard
entirely — not greyed out, not counted, not listed.

For consenting students, an officer sees: name, branch, match score, readiness
band, gap labels, application count.

An officer **cannot** see, through any endpoint: resume text, the uploaded
file, contact details, individual bullet points, generated rewrites, or
application outcomes. `routers/institution.py` constructs its responses from
scores and gap labels only; resume text is never joined into that query path.

Every dashboard load and every student detail view writes an `audit_log` row
naming the officer. Colleges receive this log on request.

This asymmetry is deliberate. The batch dashboard exists to find students who
need help early. It is not a surveillance tool, and a placement cell that could
read every resume in the batch would be one.

---

## 4. Retention

Resume text carries a `purge_after` date, defaulting to 365 days
(`GAPFOLIO_RETENTION_DAYS`). A nightly sweep deletes text and vectors past that
date (`run_retention_sweep`). Reports outlive resumes: a student keeps their
score history after the underlying resume is gone.

Job descriptions are public postings, deduplicated by content hash and shared
across users. They are not personal data and are not purged.

---

## 5. Deletion, and what survives it

`DELETE /api/privacy/account` removes the account, all resumes, all vectors,
all reports, and all consent records.

One thing survives: the de-identified `outcomes` row. It holds a status
("interview", "rejected"), the score at the time of applying, and a snapshot of
which requirements were proven or missing. It carries **no user id, no resume
text, and no job description** — after deletion its report reference is
tombstoned and cannot be resolved back to a person.

This is a deliberate design decision and students are told about it plainly at
the point of deletion. The alternative — cascading outcomes into deletion —
would mean every departing student silently degrades the model that helps the
students still using it, and would make the research dataset a function of who
happened not to leave. Keeping a record that cannot identify anyone is the more
honest trade. A student who objects to even that should decline `research_use`
at signup, in which case their outcomes are excluded from published analysis.

---

## 6. Third-party processing

| Processor | Receives | Purpose |
| --- | --- | --- |
| Anthropic | Resume text, JD text | Requirement extraction, roadmap generation |
| Embedding provider (Voyage / OpenAI) | Resume and JD chunks | Semantic matching |

Both are accessed over API, where inputs are not used for model training.
Neither receives account details — no name, email, or college is sent with the
text. Students are shown this list at consent time via `GET
/api/privacy/policy`.

---

## 7. Security

Passwords are bcrypt-hashed at cost 12, SHA-256 pre-hashed so that bcrypt's
72-byte truncation cannot make two long passwords with a shared prefix
interchangeable (`app/security.py`). Sessions are short-lived JWTs. Database
credentials and API keys come from environment variables and are absent from
the repository. All traffic must terminate TLS at the proxy.

---

## 8. Known gaps

Stated plainly, because a pilot proposal that claims none is not credible:

- **No encryption at rest beyond the database's own.** Full-disk encryption on
  the database host is the deployment's responsibility, not the application's.
- **No formal DPA template yet.** A written agreement with each partner college
  should be signed before the first upload; this document is its technical
  annexe, not a substitute.
- **No automated PII redaction inside resume text.** Text goes to processors as
  the student wrote it. Redacting names and phone numbers before the API call
  is worth building and is not built.
- **Audit log is append-only but not tamper-evident.** Hash-chaining the log is
  a small change and is not done.
- **India's DPDP Act rules were still settling as of this writing.** Consent
  granularity, deletion and purpose limitation here were designed to meet its
  direction, but the Act's final rules should be re-checked before a paid
  deployment, and a Data Protection Officer named if thresholds require one.

---

## 9. Contact

Data questions from students or colleges should reach the team directly, and a
named individual should own this document before the pilot begins. A policy
with no owner is decoration.
