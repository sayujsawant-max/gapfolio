"""Fit the matcher's thresholds to labeled data instead of picking them by eye,
and report how well the fit generalises.

Sweeps the dense/lexical blend weight and the two decision thresholds over the
hand-labeled pairs in tests/fixtures/labeled_pairs.json.

    python scripts/tune_thresholds.py                     # hash stub
    python scripts/tune_thresholds.py --provider sbert    # the real model

Three numbers are reported, because one number would flatter the fit:

  1. development  - fitted on the original resumes, scored on the same cases
  2. held-out     - the same fit, scored on cases written afterwards. This is
                    the honest estimate of how the matcher does on new resumes.
  3. cross-val    - leave-one-resume-out over every resume: fit on all the
                    others, score on the one left out.

The configuration printed at the end is fitted on every case and is what goes
into app/services/matching.py. Re-run this after any change to the embedding
provider, the chunker or the lexical layer.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--provider", default="hash", help="hash | sbert | voyage | openai")
parser.add_argument("--verbose", action="store_true", help="list every held-out disagreement")
args = parser.parse_args()

os.environ["GAPFOLIO_DATABASE_URL"] = f"sqlite:///{tempfile.mkdtemp()}/tune.db"
os.environ["GAPFOLIO_EMBEDDING_PROVIDER"] = args.provider
# The stub was tuned at 1024 dimensions; SBERT's model is 384.
os.environ["GAPFOLIO_EMBEDDING_DIM"] = "1024" if args.provider == "hash" else "384"

import numpy as np  # noqa: E402

from app.db import SessionLocal, init_db  # noqa: E402
from app.services import lexical, vectorstore  # noqa: E402
from app.services.embeddings import cosine, embed_one  # noqa: E402
from app.services.matching import _resume_chunks  # noqa: E402

DATA = json.loads((ROOT / "tests" / "fixtures" / "labeled_pairs.json").read_text(encoding="utf-8"))
DEV = set(DATA.get("_dev_resumes", []))
STATUSES = ("proven", "partial", "missing")


def build() -> list[dict]:
    """Precompute dense and lexical components once; the sweep is then free."""
    init_db()
    db = SessionLocal()
    for key, text in DATA["resumes"].items():
        vectorstore.index_document(db, "resume", key, text)

    rows = []
    for case in DATA["cases"]:
        chunks = _resume_chunks(db, case["resume"])
        idf = lexical.build_idf([c.chunk for c in chunks])
        query = f"{case['label']}. {case['quote']}"
        qvec = embed_one(query)
        scored = [
            {
                "kind": c.kind,
                "chunk": c.chunk,
                "dense": max(cosine(qvec, c.vector), 0.0),
                "lex": lexical.coverage(query, c.chunk, idf),
            }
            for c in chunks
        ]
        rows.append({**case, "chunks": scored})
    db.close()
    return rows


def relevance(row: dict, dense_w: float) -> tuple[float, float]:
    """(best over every chunk, best over chunks that can prove a requirement)."""
    best = proof = 0.0
    for c in row["chunks"]:
        rel = dense_w * c["dense"] + (1.0 - dense_w) * c["lex"]
        best = max(best, rel)
        if c["kind"] in ("achievement", "context"):
            proof = max(proof, rel)
    return best, proof


def predict(row: dict, dense_w: float, proven_t: float, partial_t: float) -> str:
    best, proof = relevance(row, dense_w)
    if proof >= proven_t:
        return "proven"
    if best >= partial_t:
        return "partial"
    return "missing"


def evaluate(rows, dense_w, proven_t, partial_t) -> tuple[float, float]:
    """(agreement, false-missing rate). A false "missing" is the costly error:
    it sends a student to learn something they already have."""
    if not rows:
        return 0.0, 0.0
    correct = false_missing = 0
    for r in rows:
        got = predict(r, dense_w, proven_t, partial_t)
        correct += got == r["expect"]
        false_missing += got == "missing" and r["expect"] != "missing"
    return correct / len(rows), false_missing / len(rows)


def fit(rows) -> tuple[float, float, float]:
    grid = []
    for dense_w in [i / 10 for i in range(0, 11)]:
        for proven_t in [i / 100 for i in range(10, 75, 2)]:
            for partial_t in [i / 100 for i in range(4, 60, 2)]:
                if partial_t >= proven_t:
                    continue
                acc, fm = evaluate(rows, dense_w, proven_t, partial_t)
                grid.append((acc, -fm, dense_w, proven_t, partial_t))
    grid.sort(reverse=True)
    return grid[0][2:]


def confusion(rows, params) -> dict:
    m = {a: {b: 0 for b in STATUSES} for a, b in zip(STATUSES, STATUSES)}
    m = {a: {b: 0 for b in STATUSES} for a in STATUSES}
    for r in rows:
        m[r["expect"]][predict(r, *params)] += 1
    return m


def print_confusion(rows, params, title) -> None:
    m = confusion(rows, params)
    print(f"\n{title}")
    print("expected \\ predicted   " + "  ".join(f"{s:>8}" for s in STATUSES))
    for a in STATUSES:
        print(f"{a:<22}" + "  ".join(f"{m[a][b]:>8}" for b in STATUSES))


def spearman(x, y) -> float:
    def ranks(v):
        order = np.argsort(v)
        r = np.empty(len(v))
        r[order] = np.arange(len(v))
        for val in set(v):  # average ranks over ties
            idx = [i for i, a in enumerate(v) if a == val]
            r[idx] = np.mean(r[idx])
        return r
    rx, ry = ranks(list(x)), ranks(list(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    rows = build()
    dev = [r for r in rows if r["resume"] in DEV]
    held = [r for r in rows if r["resume"] not in DEV]

    print(f"provider           : {args.provider}")
    print(f"labelled cases     : {len(rows)}  (development {len(dev)}, held-out {len(held)})")

    dev_params = fit(dev)
    a1, f1 = evaluate(dev, *dev_params)
    a2, f2 = evaluate(held, *dev_params)
    print(f"\nfitted on development set: dense_w={dev_params[0]} proven>={dev_params[1]} partial>={dev_params[2]}")
    print(f"  development agreement : {a1:.1%}   false 'missing' {f1:.1%}")
    print(f"  HELD-OUT agreement    : {a2:.1%}   false 'missing' {f2:.1%}   <- honest estimate")
    print_confusion(held, dev_params, "held-out confusion matrix")

    # leave-one-resume-out
    correct = fm = 0
    per_resume = {}
    for key in DATA["resumes"]:
        train = [r for r in rows if r["resume"] != key]
        test = [r for r in rows if r["resume"] == key]
        p = fit(train)
        acc, f = evaluate(test, *p)
        per_resume[key] = (acc, len(test))
        correct += round(acc * len(test))
        fm += round(f * len(test))
    print(f"\nleave-one-resume-out over {len(DATA['resumes'])} resumes: "
          f"{correct / len(rows):.1%} agreement, false 'missing' {fm / len(rows):.1%}")
    for key, (acc, n) in per_resume.items():
        print(f"  {key:<20} {acc:.0%} of {n}")

    final = fit(rows)
    acc_all, fm_all = evaluate(rows, *final)
    rho = spearman(
        [relevance(r, final[0])[1] for r in rows],
        [{"proven": 2, "partial": 1, "missing": 0}[r["expect"]] for r in rows],
    )
    print(f"\nfitted on all {len(rows)} cases: {acc_all:.1%} agreement, false 'missing' {fm_all:.1%}")
    print(f"Spearman correlation, proof relevance vs human label (0/1/2): {rho:.3f}")
    print_confusion(rows, final, "all-cases confusion matrix")

    print("\nput these in app/services/matching.py:")
    print(f"  DENSE_W    = {final[0]}")
    print(f"  LEXICAL_W  = {round(1 - final[0], 2)}")
    print(f"  PROVEN_T   = {final[1]}")
    print(f"  PARTIAL_T  = {final[2]}")

    if args.verbose:
        print("\nheld-out disagreements under the development fit:")
        for r in held:
            got = predict(r, *dev_params)
            if got != r["expect"]:
                print(f"  [{r['resume']}] {r['label']}: expected {r['expect']}, got {got}")
                print(f"      {r['why']}")


if __name__ == "__main__":
    main()
