"""The matching engine.

Pipeline per analysis:

  1. requirements  ← LLM structured extraction from the JD (heuristic fallback)
  2. evidence      ← vector search against the resume index, per requirement
  3. status        ← proven / partial / missing, from similarity *and* chunk kind
  4. score         ← weighted coverage, weights learned from real outcomes
  5. roadmap       ← LLM, given the gaps and the resume

Step 3 is the part that keeps the score honest. A skill sitting in a
comma-separated list matches strongly on similarity but is only ever "partial",
because a listed skill is a claim, not proof. That rule is why the score does
not inflate for keyword-stuffed resumes.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..models import Embedding, Resume, SkillWeight
from . import lexical
from .embeddings import cosine, embed_one
from .llm import LLMUnavailable, ROADMAP_PROMPT, REQUIREMENTS_PROMPT, complete_json
from . import vectorstore

ENGINE_VERSION = "v3-sbert-hybrid"

# Fitted, not guessed. scripts/tune_thresholds.py sweeps these against the
# hand-labeled pairs in tests/fixtures/labeled_pairs.json (76 cases, 7 resumes)
# and reports the configuration that best reproduces human judgement.
#
# Thresholds belong to the embedding model, not to the problem: cosine
# similarity has a different scale for every model. Each provider therefore
# carries its own fitted triple (dense weight, proven threshold, partial
# threshold).
#
#   sbert  all-MiniLM-L6-v2, fitted on all 76 cases: 92.1% agreement, 1.3% false
#          "missing". Fitted on the 24 development cases only, it scores 90.4%
#          on the 52 held-out cases (1.9% false "missing"); leave-one-resume-out
#          agreement is 90.8%.
#   hash   the offline test stub, fitted on all 76: 88.2% in-sample; 80.8%
#          held-out when fitted on the development cases only.
#
# voyage / openai are not calibrated; they borrow the sbert triple until
# `tune_thresholds.py --provider <name>` has been run for them.
_FITTED = {
    "sbert": (0.7, 0.34, 0.32),
    "hash": (0.7, 0.14, 0.12),
}
DENSE_W, PROVEN_T, PARTIAL_T = _FITTED.get(settings.embedding_provider, _FITTED["sbert"])
LEXICAL_W = round(1 - DENSE_W, 2)
STATUS_VALUE = {"proven": 1.0, "partial": 0.45, "missing": 0.0}


def jd_hash(text: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", text.strip().lower()).encode()).hexdigest()


# --------------------------------------------------------------------------
# 1. requirements
# --------------------------------------------------------------------------

_REQ_HEAD = re.compile(
    r"(what you'll do|responsibilities|requirements|what we're looking for|qualifications|you will)",
    re.I,
)
_BULLET = re.compile(r"^\s*(?:[-•▪◦*·–—]|\d+[.)])\s+")


def extract_requirements(jd_text: str) -> dict:
    try:
        data = complete_json(REQUIREMENTS_PROMPT.format(jd=jd_text[:6000]), max_tokens=1600)
        reqs = [r for r in data.get("requirements", []) if r.get("label")]
        if reqs:
            data["requirements"] = reqs[:12]
            return data
    except (LLMUnavailable, Exception):
        pass
    return _heuristic_requirements(jd_text)


_NICE_HEAD = re.compile(r"(nice to have|nice-to-have|preferred|bonus|good to have|desirable|plus)", re.I)
# Modal language decides mandatory versus preferred when a line states it
# outright. "A plus" beats "experience with" because it is checked first.
_NICE_WORDS = re.compile(
    r"\b(preferred|nice[ -]to[ -]have|bonus|a plus|good to have|desirable|familiarity|"
    r"exposure to|optional|ideally|advantage)\b", re.I)
_MUST_WORDS = re.compile(
    r"\b(must|required|mandatory|minimum|essential|need to|should have|"
    r"hands-on|proven|strong (?:knowledge|fundamentals|understanding))\b", re.I)


def modal_weight(line: str, default: str) -> str:
    """core / nice-to-have from the modal language in one requirement line,
    falling back to the heading-derived default when the line is silent."""
    if _NICE_WORDS.search(line):
        return "nice-to-have"
    if _MUST_WORDS.search(line):
        return "core"
    return default


def _heuristic_requirements(jd_text: str) -> dict:
    """Fallback when the model is unreachable: take the bullet lines under a
    requirements-style heading. Cruder labels, same downstream pipeline.

    Weight comes from the line's own modal language first ("must", "preferred"),
    then from the heading it sits under, then from position."""
    lines = [l.strip() for l in jd_text.splitlines() if l.strip()]
    reqs: list[dict] = []
    live = False
    for line in lines:
        if not _BULLET.match(line) and _NICE_HEAD.search(line):
            live = False
            continue
        if _REQ_HEAD.search(line):
            live = True
            continue
        if _BULLET.match(line):
            body = _BULLET.sub("", line)
            if len(body) > 12:
                reqs.append(
                    {
                        "label": " ".join(body.split()[:6]).rstrip(",.;"),
                        "jd_quote": " ".join(body.split()[:12]),
                        "weight": modal_weight(body, "core" if live else "nice-to-have"),
                        "family": "general",
                    }
                )
    title = lines[0][:120] if lines else "Target role"
    return {
        "role_title": title,
        "company": "",
        "seniority": "",
        "role_family": "general",
        "requirements": reqs[:12] or [
            {"label": "Unstructured posting", "jd_quote": title, "weight": "core", "family": "general"}
        ],
    }


# --------------------------------------------------------------------------
# 2 + 3. evidence and status
# --------------------------------------------------------------------------

@dataclass
class Matched:
    label: str
    jd_quote: str
    weight: str
    status: str
    evidence: str | None
    similarity: float
    source_kind: str


def relevance(
    requirement: str, hit: vectorstore.Hit, idf: dict[str, float] | None = None
) -> float:
    return DENSE_W * max(hit.score, 0.0) + LEXICAL_W * lexical.coverage(
        requirement, hit.chunk, idf
    )


def match_requirements(db: Session, resume_id: str, requirements: list[dict]) -> list[Matched]:
    out: list[Matched] = []
    chunks = _resume_chunks(db, resume_id)
    # Vectors from a different embedding model cannot be compared with this
    # one's. A resume indexed before the provider changed is re-indexed from its
    # stored text rather than silently scored against nonsense.
    if chunks and len(chunks[0].vector) != len(embed_one("probe")):
        resume = db.get(Resume, resume_id)
        if resume and resume.text:
            vectorstore.index_document(db, "resume", resume_id, resume.text)
            chunks = _resume_chunks(db, resume_id)
    idf = lexical.build_idf([c.chunk for c in chunks])

    for req in requirements:
        query = f"{req.get('label','')}. {req.get('jd_quote','')}"
        qvec = embed_one(query)

        # Scored exhaustively rather than through the ANN index. One resume is
        # ~100 chunks, so the full scan costs nothing, and it keeps a recall
        # miss in the vector index from silently turning a real match into a
        # gap — the failure mode that matters most here, since a false "missing"
        # sends a student off to learn something they already know. The ANN
        # index earns its keep on cross-corpus queries, not on this one.
        scored = sorted(
            (
                (c, DENSE_W * max(cosine(qvec, c.vector), 0.0)
                    + LEXICAL_W * lexical.coverage(query, c.chunk, idf))
                for c in chunks
            ),
            key=lambda p: p[1],
            reverse=True,
        )
        best, best_rel = scored[0] if scored else (None, 0.0)

        # Only an achievement or a described responsibility can *prove* a
        # requirement. A skill sitting in a comma-separated list is a claim.
        proof = next(
            (
                (h, rel)
                for h, rel in scored
                if h.kind in ("achievement", "context") and rel >= PROVEN_T
            ),
            None,
        )

        if proof:
            hit, rel = proof
            status, evidence, sim, kind = "proven", hit.chunk, rel, hit.kind
        elif best is not None and best_rel >= PARTIAL_T:
            status, evidence, sim, kind = "partial", best.chunk, best_rel, best.kind
        else:
            status, evidence, sim, kind = "missing", None, best_rel, ""

        out.append(
            Matched(
                label=req.get("label", ""),
                jd_quote=req.get("jd_quote", ""),
                weight=req.get("weight", "core"),
                status=status,
                evidence=_trim(evidence),
                similarity=round(sim, 3),
                source_kind=kind,
            )
        )
    return out


def _resume_chunks(db: Session, resume_id: str) -> list[Embedding]:
    return list(
        db.scalars(
            select(Embedding).where(
                Embedding.owner_type == "resume", Embedding.owner_id == resume_id
            )
        ).all()
    )


def _trim(text: str | None, words: int = 16) -> str | None:
    if not text:
        return None
    parts = text.split()
    return " ".join(parts[:words]) + ("…" if len(parts) > words else "")


# --------------------------------------------------------------------------
# 4. score
# --------------------------------------------------------------------------

def load_weights(db: Session, role_family: str) -> dict[str, float]:
    rows = db.scalars(
        select(SkillWeight).where(SkillWeight.role_family.in_([role_family, "general"]))
    ).all()
    weights: dict[str, float] = {}
    for row in rows:
        key = row.skill.lower()
        # A family-specific weight beats the general one.
        if row.role_family == role_family or key not in weights:
            weights[key] = row.weight
    return weights


def score(matches: list[Matched], weights: dict[str, float]) -> tuple[float, str]:
    total = earned = 0.0
    for m in matches:
        base = 1.0 if m.weight == "core" else 0.4
        learned = weights.get(m.label.lower(), 1.0)
        w = base * learned
        total += w
        earned += w * STATUS_VALUE[m.status]
    pct = round(100 * earned / total, 1) if total else 0.0
    readiness = "Ready" if pct >= 75 else "Close" if pct >= 55 else "Not yet"
    return pct, readiness


def headline(matches: list[Matched], role: str) -> str:
    missing_core = [m.label for m in matches if m.status == "missing" and m.weight == "core"]
    unproven = [m.label for m in matches if m.status == "partial"]
    if missing_core:
        head = ", ".join(missing_core[:3])
        return f"Nothing on the page proves {head} — the core of what {role} is asking for."
    if unproven:
        return (
            f"You claim {', '.join(unproven[:3])} but nothing backs it up. "
            "Listed skills read as unproven."
        )
    return "Your resume covers the posting. The remaining work is sharpening how it reads."


# --------------------------------------------------------------------------
# 5. roadmap
# --------------------------------------------------------------------------

#
# Gap ranking (Stage 4). Impact is how much closing the gap would add to the
# score; effort is a rough, explainable estimate of how long closing it takes.
# Both are deterministic, so the order of the roadmap never depends on the LLM.
#
#   impact   = requirement weight x learned weight x (1 - status value)
#   priority = impact / sqrt(effort_days)
#
# The square root keeps a large core gap above a trivial nice-to-have while still
# pulling quick wins (a bullet to rewrite) ahead of slow ones of equal impact.

EFFORT_DAYS = {"rewrite": 1, "project": 4, "tool": 10, "concept": 14}
PRIORITY_HIGH, PRIORITY_MEDIUM = 0.30, 0.15


def _effort(m: Matched) -> tuple[str, int]:
    if m.status == "partial":
        # Evidence exists. If it is only a list entry the candidate needs a
        # project behind it; otherwise the bullet just needs rewriting.
        return ("project", EFFORT_DAYS["project"]) if m.source_kind == "skill_listed" else (
            "rewrite", EFFORT_DAYS["rewrite"])
    named_tool = bool(lexical.tokens(m.label) & set(lexical.ALIASES))
    return ("course", EFFORT_DAYS["tool"] if named_tool else EFFORT_DAYS["concept"])


def rank_gaps(matches: list[Matched], weights: dict[str, float] | None = None) -> list[dict]:
    weights = weights or {}
    ranked = []
    for m in matches:
        if m.status == "proven":
            continue
        base = 1.0 if m.weight == "core" else 0.4
        impact = base * weights.get(m.label.lower(), 1.0) * (1.0 - STATUS_VALUE[m.status])
        fix_type, days = _effort(m)
        priority = impact / days ** 0.5
        ranked.append({
            "label": m.label, "status": m.status, "weight": m.weight,
            "evidence": m.evidence, "source_kind": m.source_kind,
            "impact": round(impact, 3), "effort_days": days, "fix_type": fix_type,
            "priority_score": round(priority, 3),
            "priority": "high" if priority >= PRIORITY_HIGH else "medium" if priority >= PRIORITY_MEDIUM else "low",
        })
    ranked.sort(key=lambda g: g["priority_score"], reverse=True)
    return ranked


def build_roadmap(
    role: str,
    matches: list[Matched],
    weak_bullets: list[str] | None = None,
    weights: dict[str, float] | None = None,
) -> dict:
    """Ranked roadmap, explained by the LLM when one is configured.

    Grounding (Stage 1, section 3.3): the model receives the ranked gaps and the
    flagged weak bullets, never the resume. Whatever it returns is then checked,
    and anything it was not given a basis for is dropped.
    """
    ranked = rank_gaps(matches, weights)[:5]
    bullets = [_strip_bullet(b) for b in (weak_bullets or [])][:6]
    fallback = _fallback_roadmap(ranked, bullets)
    if not ranked:
        return {"roadmap": [], "rewrites": fallback["rewrites"], "questions": []}

    gap_lines = "\n".join(
        f"- {g['label']} [{g['status']}, {g['weight']}] — resume shows: {g['evidence'] or 'nothing'}"
        for g in ranked
    )
    weak_lines = "\n".join(f"- {b}" for b in bullets) or "- (none flagged)"
    try:
        data = complete_json(
            ROADMAP_PROMPT.format(role=role, gap_lines=gap_lines, weak_lines=weak_lines),
            max_tokens=2000,
        )
    except Exception:
        return fallback
    return _ground(data, ranked, bullets, fallback)


_NUM = re.compile(r"\d+(?:\.\d+)?")
_BULLET_PREFIX = re.compile(r"^\s*(?:[-•▪◦*·–—]|\d+[.)])\s+")


def _strip_bullet(text: str) -> str:
    return _BULLET_PREFIX.sub("", text).strip()


def _ground(data: dict, ranked: list[dict], bullets: list[str], fallback: dict) -> dict:
    """Keep only what the structured input supports.

    - a roadmap item must name one of the ranked gaps, and takes its priority,
      effort and fix type from the deterministic ranking, not from the model;
    - a rewrite's "before" must be one of the flagged bullets, and its "after"
      may not contain a number that the original did not.
    """
    by_label = {g["label"].lower(): g for g in ranked}
    roadmap = []
    for item in data.get("roadmap") or []:
        gap = by_label.get(str(item.get("skill", "")).lower())
        if not gap:
            continue
        roadmap.append({
            "skill": gap["label"], "priority": gap["priority"],
            "priority_score": gap["priority_score"], "effort": f"{gap['effort_days']} days",
            "effort_days": gap["effort_days"], "fix_type": gap["fix_type"],
            "action": str(item.get("action", "")), "proof": str(item.get("proof", "")),
        })
    covered = {r["skill"] for r in roadmap}
    roadmap += [r for r in fallback["roadmap"] if r["skill"] not in covered]
    roadmap.sort(key=lambda r: r["priority_score"], reverse=True)

    rewrites = []
    for rw in data.get("rewrites") or []:
        before = _strip_bullet(str(rw.get("before", "")))
        after = str(rw.get("after", ""))
        if not any(before == b or (len(before) > 20 and b.startswith(before)) for b in bullets):
            continue
        if set(_NUM.findall(after)) - set(_NUM.findall(before)):
            continue
        rewrites.append({"before": before, "after": after, "why": str(rw.get("why", ""))})

    questions = [str(q) for q in (data.get("questions") or [])][:3]
    return {"roadmap": roadmap, "rewrites": rewrites or fallback["rewrites"], "questions": questions}


def _fallback_roadmap(ranked: list[dict], bullets: list[str]) -> dict:
    """The whole roadmap with no LLM: templated, deterministic, and grounded by
    construction, since every sentence is built from a gap label."""
    roadmap = []
    for g in ranked:
        if g["fix_type"] == "rewrite":
            action = (f"Your resume touches {g['label']} but the evidence is thin. Rewrite that bullet to "
                      "say what you built and what changed.")
        elif g["fix_type"] == "project":
            action = (f"You list {g['label']} but nothing demonstrates it. Build one small project that "
                      "uses it end to end and describe the outcome, not the tool.")
        else:
            action = f"Close {g['label']} with a short course, then apply it in a project you can point at."
        roadmap.append({
            "skill": g["label"], "priority": g["priority"], "priority_score": g["priority_score"],
            "effort": f"{g['effort_days']} days", "effort_days": g["effort_days"],
            "fix_type": g["fix_type"], "action": action,
            "proof": f"Built a project using {g['label']} that [what it does], measured at [N].",
        })

    rewrites = []
    for b in bullets[:3]:
        words = b.split()
        rest = " ".join(words[1:]) if words and words[0].lower().strip(",") in WEAK_VERBS_FALLBACK else b
        rewrites.append({
            "before": b,
            "after": f"[Strong verb] {rest} — name the tools used and the measurable result [N].",
            "why": "Opens with a verb that describes presence, not contribution, and has no number.",
        })
    return {"roadmap": roadmap, "rewrites": rewrites, "questions": []}


from .parsing import WEAK_VERBS as WEAK_VERBS_FALLBACK  # noqa: E402
