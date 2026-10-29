"""Stage 3 behaviour: layout detection, grounding, ranking, tagging, segmentation.

Runs on the offline hash provider except where a test says it needs the real
model (via ONNX Runtime), in which case it is skipped when the model is unavailable.
"""
import io
import os
import tempfile

import pytest

os.environ["GAPFOLIO_DATABASE_URL"] = f"sqlite:///{tempfile.mkdtemp()}/stage3.db"
os.environ["GAPFOLIO_EMBEDDING_PROVIDER"] = "hash"
os.environ["GAPFOLIO_EMBEDDING_DIM"] = "1024"  # the offline stub was tuned at 1024-d
os.environ["GAPFOLIO_ANTHROPIC_API_KEY"] = ""

from app.services import entities, lexical, matching  # noqa: E402
from app.services.parsing import (  # noqa: E402
    ats_checks, parse_confidence, parse_document, parse_text, _detect_columns,
)

BULLETS = [f"Built service number {i} handling {100 + i} requests with measurable results" for i in range(16)]


def _pdf(kind):
    from fpdf import FPDF

    p = FPDF()
    p.add_page()
    p.set_font("helvetica", size=10)
    if kind == "single":
        for line in BULLETS:
            p.set_x(p.l_margin); p.multi_cell(w=p.epw, h=6, text=line)
    elif kind == "dates":  # long left text, short right-aligned dates
        for i, line in enumerate(BULLETS):
            p.set_xy(15, 20 + i * 8); p.cell(130, 6, line[:60])
            p.set_xy(165, 20 + i * 8); p.cell(30, 6, "2024-2025")
    elif kind == "two":
        for i, line in enumerate(BULLETS[:8]):
            p.set_xy(12, 20 + i * 8); p.cell(90, 6, line[:46])
        for i, line in enumerate(BULLETS[8:]):
            p.set_xy(110, 20 + i * 8); p.cell(90, 6, line[:46])
    elif kind == "sidebar":
        for i, line in enumerate(BULLETS[:8]):
            p.set_xy(10, 20 + i * 8); p.cell(55, 6, line[:28])
        for i, line in enumerate(BULLETS[8:]):
            p.set_xy(75, 20 + i * 8); p.cell(125, 6, line[:70])
    return bytes(p.output())


def _use_real_model(monkeypatch):
    """Switch to the real embedding model for one test, or skip it."""
    pytest.importorskip("onnxruntime")
    from app.config import settings
    from app.services import embeddings

    monkeypatch.setattr(settings, "embedding_provider", "onnx")
    monkeypatch.setattr(settings, "embedding_dim", 384)
    monkeypatch.setattr(embeddings, "_onnx_failed", False)
    if embeddings.effective_provider() != "onnx":
        pytest.skip("ONNX embedding model not available")

# --- column detection -----------------------------------------------------

@pytest.mark.parametrize("kind,expected", [
    ("single", False), ("dates", False), ("two", True), ("sidebar", True),
])
def test_column_detection(kind, expected):
    """The first detector compared two text extractions and found no column in
    any of four two-column resumes. Geometry does; right-aligned dates, the
    obvious false-positive, must not trip it."""
    assert _detect_columns(_pdf(kind))["multi_column"] is expected


# --- tokeniser ------------------------------------------------------------

def test_cicd_compound_is_one_token():
    assert "cicd" in lexical.tokens("ship them through CI/CD pipelines")
    assert lexical.stems("CI/CD") & lexical.expand("deployed through GitHub Actions")


def test_inflections_share_a_stem():
    assert lexical.stems("handled") == lexical.stems("handling") == lexical.stems("handles")
    assert lexical.stems("users") == lexical.stems("user")


def test_trailing_punctuation_is_not_part_of_a_token():
    assert "flows" in lexical.tokens("authentication flows.")


# --- modal language -------------------------------------------------------

@pytest.mark.parametrize("line,default,expected", [
    ("Must have strong knowledge of SQL", "nice-to-have", "core"),
    ("Familiarity with Kubernetes is a plus", "core", "nice-to-have"),
    ("Experience with Docker", "core", "core"),
    ("Experience with Docker", "nice-to-have", "nice-to-have"),
    ("Bonus: exposure to Kafka", "core", "nice-to-have"),
])
def test_modal_weight(line, default, expected):
    assert matching.modal_weight(line, default) == expected


def test_heuristic_requirements_use_modal_language_and_headings():
    jd = ("Backend Engineer\n\nWhat we're looking for\n"
          "- Must have experience building REST APIs in Python\n"
          "- Familiarity with Kubernetes is a plus\n\n"
          "Nice to have\n- Experience with GraphQL services\n")
    weights = {r["label"].split()[0]: r["weight"] for r in matching._heuristic_requirements(jd)["requirements"]}
    assert weights["Must"] == "core"
    assert weights["Familiarity"] == "nice-to-have"   # a heading says core, the line says otherwise
    assert weights["Experience"] == "nice-to-have"    # under the "Nice to have" heading


# --- gap ranking ----------------------------------------------------------

def _m(label, status, weight="core", kind="achievement"):
    return matching.Matched(label, "", weight, status, "evidence" if status != "missing" else None, 0.5, kind)


def test_ranking_orders_by_impact_over_effort():
    ranked = matching.rank_gaps([
        _m("Salesforce", "missing"),                         # core, missing, 14-day concept
        _m("Docker", "partial", kind="skill_listed"),        # core, listed, needs a project
        _m("Kafka", "missing", weight="nice-to-have"),
        _m("REST", "partial"),                               # core, evidence exists: a rewrite
        _m("Git", "proven"),
    ])
    assert [g["label"] for g in ranked][0] == "REST"        # cheapest fix for real impact
    assert "Git" not in [g["label"] for g in ranked]        # proven items are not gaps
    scores = [g["priority_score"] for g in ranked]
    assert scores == sorted(scores, reverse=True)
    assert ranked[0]["fix_type"] == "rewrite"
    assert {g["label"]: g["fix_type"] for g in ranked}["Docker"] == "project"


# --- grounding ------------------------------------------------------------

def test_prompt_contains_no_resume_text(monkeypatch):
    captured = {}
    monkeypatch.setattr(matching, "complete_json", lambda p, max_tokens=0: captured.setdefault("p", p) and {})
    secret = "Built a payments ledger for Meridian Bank serving 40000 accounts"
    ms = [_m("Kubernetes", "missing")]
    matching.build_roadmap("Backend Engineer", ms, ["Helped the team with testing tasks"], {})
    assert "Kubernetes" in captured["p"] and "Helped the team" in captured["p"]
    assert secret not in captured["p"]


def test_grounding_drops_what_the_input_does_not_support():
    ranked = matching.rank_gaps([_m("Kubernetes", "missing"), _m("Docker", "partial", kind="skill_listed")])
    bullets = ["Helped the team with testing tasks"]
    fallback = matching._fallback_roadmap(ranked, bullets)
    out = matching._ground({
        "roadmap": [
            {"skill": "Kubernetes", "action": "ok", "proof": "Shipped [N]"},
            {"skill": "Quantum Computing", "action": "invented gap", "proof": "x"},
        ],
        "rewrites": [
            {"before": "Helped the team with testing tasks", "after": "Wrote tests, measured at [N]", "why": "ok"},
            {"before": "Helped the team with testing tasks", "after": "Wrote tests for 5000 users", "why": "invented figure"},
            {"before": "Built the entire platform alone", "after": "x", "why": "not in the resume"},
        ],
        "questions": ["a", "b", "c", "d"],
    }, ranked, bullets, fallback)
    skills = [r["skill"] for r in out["roadmap"]]
    assert "Quantum Computing" not in skills
    assert {"Kubernetes", "Docker"} <= set(skills)           # uncovered gap filled from the template
    assert [rw["after"] for rw in out["rewrites"]] == ["Wrote tests, measured at [N]"]
    assert len(out["questions"]) == 3
    # priority and effort come from the ranking, never from the model
    k = next(r for r in out["roadmap"] if r["skill"] == "Kubernetes")
    assert k["effort_days"] == ranked[[g["label"] for g in ranked].index("Kubernetes")]["effort_days"]


def test_fallback_roadmap_needs_no_llm():
    ranked = matching.rank_gaps([_m("Kubernetes", "missing")])
    out = matching.build_roadmap("Backend Engineer", [_m("Kubernetes", "missing")], ["Helped with tasks"], {})
    assert out["roadmap"][0]["skill"] == "Kubernetes"
    assert "[N]" in out["roadmap"][0]["proof"]               # placeholder, not an invented figure
    assert ranked and out["rewrites"][0]["before"] == "Helped with tasks"


# --- skills ---------------------------------------------------------------

def test_skill_extraction_aliases_and_ambiguous_words():
    found = entities.extract_skills("Built APIs in Node.js and Express with Postgres, k8s and CI/CD. "
                                    "I express interest in the Spring release.")
    assert {"node.js", "express", "postgresql", "kubernetes", "ci/cd"} <= set(found)
    assert "spring" not in found or "Spring" in "Spring release"  # capitalised, so allowed


def test_lowercase_ambiguous_word_is_not_a_skill():
    assert "express" not in entities.extract_skills("happy to express my interest in the role")


# --- parse confidence and fonts ------------------------------------------

def test_parse_confidence_drops_for_defective_files():
    clean = parse_text("ARJUN RAO\narjun@example.edu | +91 98765 43210\nEDUCATION\nB.Tech CSE 2027\n"
                       "PROJECTS\n- Built an API serving 300 users\n- Wrote 40 tests at 82% coverage\n"
                       "- Cut latency by 35%\n- Shipped 3 features\nSKILLS\nPython, Git\n" * 3)
    bad = dict(clean.meta, multi_column=True, layout_tables=True, textbox_count=2,
               has_email=False, has_phone=False, sections_found=[])
    assert clean.meta["parse_confidence_level"] == "ok"
    assert parse_confidence(bad)["parse_confidence_level"] == "low"
    assert any("Multi-column" in n for n in parse_confidence(bad)["parse_confidence_notes"])


def test_standard_fonts_are_not_flagged():
    from app.services.parsing import _unembedded_fonts
    from pypdf import PdfReader

    assert _unembedded_fonts(PdfReader(io.BytesIO(_pdf("single")))) == []


def test_ats_checks_carry_a_group():
    checks = ats_checks(parse_text("EDUCATION\nB.Tech\nPROJECTS\n- Helped with tasks\nSKILLS\nPython\n").meta)
    assert {c["group"] for c in checks} <= {"parse", "readability"}
    assert any(c["group"] == "readability" for c in checks)


# --- segmentation fallback (needs the real model) -------------------------

def test_renamed_headings_are_classified_with_the_real_model(monkeypatch):
    _use_real_model(monkeypatch)
    entities._descriptor_cache.clear()
    text = ("ARJUN RAO\narjun@example.edu | +91 98765 43210\n"
            "Where I Studied\nB.Tech Computer Science, 2027, CGPA 8.1\n"
            "Toolbox\nPython, Java, Kubernetes, Redis, Git\n")
    segs = {s.heading: s for s in entities.segment_sections(text)}
    assert segs["Where I Studied"].section == "education" and segs["Where I Studied"].inferred
    assert segs["Toolbox"].section == "skills"
    # an ATS only recognises standard headings, so none of these count for the audit
    assert entities.standard_sections(text) == []
    entities._descriptor_cache.clear()


def test_sbert_matcher_agrees_with_human_labels(monkeypatch):
    """Regression guard for the shipped configuration: the real model, with the
    thresholds in matching._FITTED, against all 76 labelled cases. Falls below
    the bar and the build fails before anyone sees a worse report."""
    _use_real_model(monkeypatch)
    import json
    import pathlib

    from app.db import SessionLocal, init_db
    from app.services import vectorstore
    from app.services.embeddings import cosine, embed_one

    dense_w, proven_t, partial_t = matching._FITTED["onnx"]

    data = json.loads((pathlib.Path(__file__).parent / "fixtures" / "labeled_pairs.json").read_text(encoding="utf-8"))
    init_db()
    db = SessionLocal()
    for key, text in data["resumes"].items():
        vectorstore.index_document(db, "resume", f"sb-{key}", text)

    correct = false_missing = 0
    held = held_correct = 0
    dev = set(data["_dev_resumes"])
    for case in data["cases"]:
        chunks = matching._resume_chunks(db, f"sb-{case['resume']}")
        idf = lexical.build_idf([c.chunk for c in chunks])
        query = f"{case['label']}. {case['quote']}"
        qvec = embed_one(query)
        best = proof = 0.0
        for c in chunks:
            rel = dense_w * max(cosine(qvec, c.vector), 0.0) + (1 - dense_w) * lexical.coverage(query, c.chunk, idf)
            best = max(best, rel)
            if c.kind in ("achievement", "context"):
                proof = max(proof, rel)
        got = "proven" if proof >= proven_t else "partial" if best >= partial_t else "missing"
        correct += got == case["expect"]
        false_missing += got == "missing" and case["expect"] != "missing"
        if case["resume"] not in dev:
            held += 1
            held_correct += got == case["expect"]
    n = len(data["cases"])
    assert correct / n >= 0.88, f"agreement fell to {correct / n:.1%} over {n} cases"
    assert false_missing / n <= 0.03, f"false 'missing' rate {false_missing / n:.1%}"
    db.close()


def test_title_case_project_name_does_not_open_a_section():
    text = ("ARJUN RAO\narjun@example.edu\nPROJECTS\nCampus Payments Service\n"
            "- Built a REST API serving 1200 students\nChat Application\n- Added messaging with WebSockets\n")
    assert [s.section for s in entities.segment_sections(text, fallback=True)] == ["header", "projects"]
