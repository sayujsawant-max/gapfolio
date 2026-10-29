"""The feedback loop.

Students report what happened after they applied. Calibration turns those
reports into per-skill weights, so a requirement that actually predicts a
callback counts for more than one that doesn't. Without this, the match score
is an opinion; with it, the score is a claim you can check.

Everything here reads `outcomes`, which is de-identified — so calibration keeps
working after a student deletes their account.
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Outcome, SkillWeight

CALLBACK = {"screened", "interview", "offer"}
MIN_SAMPLES = 8          # below this, a skill keeps the neutral weight of 1.0
MAX_ADJUST = 0.6         # weights stay inside [0.4, 1.6]


def _is_callback(status: str) -> bool:
    return status in CALLBACK


def recalibrate(db: Session, role_family: str = "general") -> dict:
    """Recompute skill weights from every recorded outcome. Idempotent —
    designed to run nightly."""
    outcomes = db.scalars(select(Outcome)).all()
    # skill -> [proven_callbacks, proven_total, notproven_callbacks, notproven_total]
    tally: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])

    for o in outcomes:
        hit = _is_callback(o.status)
        for req in o.requirement_snapshot or []:
            label = (req.get("label") or "").strip().lower()
            if not label:
                continue
            proven = req.get("status") == "proven"
            base = 0 if proven else 2
            tally[label][base + 1] += 1
            if hit:
                tally[label][base] += 1

    updated = 0
    for skill, (pc, pt, nc, nt) in tally.items():
        n = pt + nt
        if n < MIN_SAMPLES or pt == 0 or nt == 0:
            continue
        # Laplace-smoothed callback rates.
        p_proven = (pc + 1) / (pt + 2)
        p_not = (nc + 1) / (nt + 2)
        lift = p_proven - p_not
        # Confidence shrinks the adjustment on small samples.
        confidence = min(1.0, math.log1p(n) / math.log1p(60))
        weight = 1.0 + max(-MAX_ADJUST, min(MAX_ADJUST, lift * 2.0)) * confidence

        row = db.scalar(
            select(SkillWeight).where(
                SkillWeight.skill == skill, SkillWeight.role_family == role_family
            )
        )
        if row is None:
            row = SkillWeight(skill=skill, role_family=role_family)
            db.add(row)
        row.weight = round(weight, 3)
        row.lift = round(lift, 3)
        row.sample_size = n
        row.updated_at = datetime.now(timezone.utc)
        updated += 1

    db.commit()
    return {"skills_evaluated": len(tally), "skills_updated": updated, "outcomes": len(outcomes)}


def evaluate(db: Session) -> dict:
    """Is the score actually predictive? This is the RQ2 measurement — run it
    on the pilot data and the answer goes straight into the paper."""
    outcomes = db.scalars(select(Outcome)).all()
    if len(outcomes) < 5:
        return {"n": len(outcomes), "verdict": "Not enough outcomes yet.", "bands": []}

    bands = [(0, 40), (40, 55), (55, 70), (70, 85), (85, 101)]
    rows = []
    for lo, hi in bands:
        group = [o for o in outcomes if lo <= o.score_at_apply < hi]
        if not group:
            continue
        hits = sum(1 for o in group if _is_callback(o.status))
        rows.append(
            {
                "band": f"{lo}-{hi - 1}",
                "n": len(group),
                "callback_rate": round(hits / len(group), 3),
            }
        )

    xs = [o.score_at_apply for o in outcomes]
    ys = [1.0 if _is_callback(o.status) else 0.0 for o in outcomes]
    r = _pearson(xs, ys)
    verdict = (
        "Score tracks callbacks." if r > 0.25
        else "Weak relationship — the score needs work." if r > 0.05
        else "No relationship yet. Do not claim predictiveness."
    )
    return {"n": len(outcomes), "point_biserial_r": round(r, 3), "verdict": verdict, "bands": rows}


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else 0.0
