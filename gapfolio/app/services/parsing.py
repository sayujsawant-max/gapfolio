"""Resume parsing.

Two jobs, done in one pass:

1. Get clean text out of a PDF, DOCX or TXT upload.
2. Record *structural* signals while the file is still a file — whether it was
   a scan, whether it used a two-column layout, whether the contact block sits
   in a header. These are exactly the things that break an ATS, and they are
   invisible once you flatten the document to a string. Running the ATS check
   on extracted text alone is how the earlier prototype under-reported risk.

Nothing is written to disk. Bytes are parsed in memory and dropped.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field, asdict
from typing import Any

from .entities import extract_skills, segment_sections, standard_sections

MAX_CHARS = 40_000
LOW_CONFIDENCE = 0.6

WEAK_VERBS = {
    "helped", "assisted", "worked", "participated", "involved", "responsible",
    "handled", "supported", "made", "did", "learned", "familiar", "exposure",
    "attended", "contributed",
}

SECTION_WORDS = {
    "education", "experience", "work experience", "projects", "skills",
    "internship", "internships", "certifications", "achievements",
    "positions", "publications", "summary", "objective", "activities",
}

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s-]?)?\d{5}[\s-]?\d{5}|\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}")
URL_RE = re.compile(r"(https?://\S+|(?:github|linkedin|gitlab)\.com/\S+)", re.I)
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:%|x|k|l|lakh|cr|users?|users|ms|s\b|hrs?|days?|weeks?|months?)|\b\d{2,}\b|\d+(?:\.\d+)?%")
BULLET_RE = re.compile(r"^\s*(?:[-•▪◦*·–—]|\d+[.)])\s+")


class ParseError(ValueError):
    pass


@dataclass
class ParseResult:
    text: str
    source: str
    meta: dict[str, Any] = field(default_factory=dict)

    def dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# format-specific extraction
# --------------------------------------------------------------------------

def _parse_pdf(data: bytes) -> tuple[str, dict]:
    from pdfminer.high_level import extract_text_to_fp
    from pdfminer.layout import LAParams

    out2 = io.StringIO()
    extract_text_to_fp(io.BytesIO(data), out2, laparams=LAParams(), codec=None)
    natural = out2.getvalue()

    meta: dict[str, Any] = {"format": "pdf"}
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        meta["pages"] = len(reader.pages)
        images = sum(len(getattr(p, "images", []) or []) for p in reader.pages[:5])
        meta["embedded_images"] = images
        meta["unembedded_fonts"] = _unembedded_fonts(reader)
    except Exception:
        meta["pages"] = None

    chars = len(re.sub(r"\s", "", natural))
    # A page of real text is >600 chars. Far less means a scan or an image export.
    meta["is_scanned"] = bool(chars < 200 * max(meta.get("pages") or 1, 1))
    meta["needs_ocr"] = meta["is_scanned"]
    if not meta["is_scanned"]:
        meta.update(_detect_columns(data))

    if meta["is_scanned"]:
        raise ParseError(
            "This PDF has no text layer — it is a scan or an image export. "
            "An ATS reads nothing from it. Export a text PDF from Word or Google Docs."
        )
    return natural, meta


def _detect_columns(data: bytes) -> dict[str, Any]:
    """Column detection from text-line geometry.

    The first version compared two text extractions and flagged a difference in
    reading order. It detected nothing: both extractions interleave a two-column
    page identically, so there was never a difference to find. Geometry is the
    reliable signal. A column is a group of substantive lines sharing a left
    edge; two columns are two such groups that sit side by side (the left one
    ends before the right one starts) and share vertical extent.

    Short lines (dates, right-aligned locations) are ignored, and the
    left-edge tolerance absorbs bullet indents, so a single-column resume with
    tab-aligned dates is not mistaken for two columns.
    """
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LAParams, LTTextContainer, LTTextLine

    best = 1
    try:
        for layout in extract_pages(io.BytesIO(data), laparams=LAParams(), maxpages=3):
            width = float(layout.width)
            lines = [
                (ln.x0, ln.x1, ln.y0, ln.y1)
                for box in layout if isinstance(box, LTTextContainer)
                for ln in box if isinstance(ln, LTTextLine) and len(ln.get_text().strip()) >= 12
            ]
            best = max(best, _count_columns(lines, width))
    except Exception:
        return {"multi_column": False, "column_count": 1}
    return {"multi_column": best >= 2, "column_count": best}


def _count_columns(lines: list[tuple[float, float, float, float]], width: float) -> int:
    if len(lines) < 8 or width <= 0:
        return 1
    tol = 0.05 * width
    groups: list[list[tuple[float, float, float, float]]] = []
    for ln in sorted(lines, key=lambda l: l[0]):
        if groups and ln[0] - groups[-1][0][0] <= tol:
            groups[-1].append(ln)
        else:
            groups.append([ln])
    groups = [g for g in groups if len(g) >= 4]
    if len(groups) < 2:
        return 1

    def median(vals):
        s = sorted(vals)
        return s[len(s) // 2]

    columns = [groups[0]]
    for g in groups[1:]:
        left = columns[-1]
        apart = g[0][0] - left[0][0] >= 0.25 * width
        beside = median(l[1] for l in left) <= median(l[0] for l in g) + tol
        shared = sum(1 for a in g for b in left if a[2] < b[3] and a[3] > b[2])
        if apart and beside and shared >= 3:
            columns.append(g)
    return len(columns)


_STANDARD_FONTS = {"Helvetica", "Times", "Courier", "Symbol", "ZapfDingbats", "Arial"}


def _unembedded_fonts(reader) -> list[str]:
    """Fonts the file uses but does not carry. A viewer substitutes them, and
    text extraction that depends on the glyph map can come out garbled. The 14
    standard fonts are exempt: every reader has them."""
    missing: set[str] = set()
    for page in reader.pages[:3]:
        try:
            fonts = page["/Resources"].get_object().get("/Font", {}).get_object()
        except Exception:
            continue
        for ref in fonts.values():
            font = ref.get_object()
            name = str(font.get("/BaseFont", "")).lstrip("/")
            target = font
            if font.get("/Subtype") == "/Type0":
                try:
                    target = font["/DescendantFonts"][0].get_object()
                except Exception:
                    continue
            descriptor = target.get("/FontDescriptor")
            if descriptor is None:
                continue  # standard-14 fonts carry no descriptor
            d = descriptor.get_object()
            if not any(k in d for k in ("/FontFile", "/FontFile2", "/FontFile3")):
                if not any(name.split("+")[-1].startswith(s) for s in _STANDARD_FONTS):
                    missing.add(name.split("+")[-1] or "unknown")
    return sorted(missing)


def _parse_docx(data: bytes) -> tuple[str, dict]:
    import docx  # python-docx

    doc = docx.Document(io.BytesIO(data))
    parts: list[str] = []
    meta: dict[str, Any] = {"format": "docx"}

    # Contact details parked in a header are invisible to most parsers.
    header_text = []
    for section in doc.sections:
        for p in section.header.paragraphs:
            if p.text.strip():
                header_text.append(p.text.strip())
    meta["contact_in_header"] = bool(EMAIL_RE.search(" ".join(header_text)))
    parts.extend(header_text)

    for p in doc.paragraphs:
        if p.text.strip():
            style = (p.style.name or "").lower()
            parts.append(p.text.strip())
            if "heading" in style:
                meta.setdefault("styled_headings", []).append(p.text.strip())

    tables = doc.tables
    meta["table_count"] = len(tables)
    for t in tables:
        for row in t.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append("  ".join(dict.fromkeys(cells)))

    # Tables are the single most common cause of scrambled ATS output.
    meta["layout_tables"] = len(tables) > 0
    meta["textbox_count"] = len(doc.element.body.findall(".//{*}txbxContent"))
    return "\n".join(parts), meta


def _parse_txt(data: bytes) -> tuple[str, dict]:
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(enc), {"format": "txt"}
        except UnicodeDecodeError:
            continue
    raise ParseError("Could not decode this file as text.")


# --------------------------------------------------------------------------
# structural signals
# --------------------------------------------------------------------------

def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:MAX_CHARS]


def structural_signals(text: str) -> dict[str, Any]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    bullets = [l for l in lines if BULLET_RE.match(l)]
    quantified = [b for b in bullets if NUMBER_RE.search(b)]
    weak = [b for b in bullets if _first_word(b) in WEAK_VERBS]
    segments = segment_sections(text)
    found_sections = standard_sections(text)
    return {
        "sections_inferred": sorted({s.section for s in segments if s.inferred}),
        "skills_detected": extract_skills(text),
        "char_count": len(text),
        "line_count": len(lines),
        "bullet_count": len(bullets),
        "quantified_bullets": len(quantified),
        "quantified_ratio": round(len(quantified) / len(bullets), 2) if bullets else 0.0,
        "weak_verb_bullets": [b[:120] for b in weak][:12],
        "sections_found": found_sections,
        "has_email": bool(EMAIL_RE.search(text)),
        "has_phone": bool(PHONE_RE.search(text)),
        "links": list(dict.fromkeys(URL_RE.findall(text)))[:8],
        "has_code_link": bool(re.search(r"github|gitlab", text, re.I)),
    }


def _first_word(bullet: str) -> str:
    body = BULLET_RE.sub("", bullet).strip().lower()
    return re.split(r"\W+", body)[0] if body else ""


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------

def parse_document(data: bytes, filename: str) -> ParseResult:
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        raw, meta = _parse_pdf(data)
        source = "pdf"
    elif name.endswith(".docx"):
        raw, meta = _parse_docx(data)
        source = "docx"
    elif name.endswith((".txt", ".md")):
        raw, meta = _parse_txt(data)
        source = "txt"
    elif name.endswith(".doc"):
        raise ParseError("Legacy .doc isn't supported. Save as .docx or PDF and upload again.")
    else:
        raise ParseError("Upload a PDF, DOCX or TXT file.")

    text = _clean(raw)
    if len(re.sub(r"\s", "", text)) < 150:
        raise ParseError("Almost no text came out of this file. Check that it isn't an image.")

    meta.update(structural_signals(text))
    meta.update(parse_confidence(meta))
    return ParseResult(text=text, source=source, meta=meta)


def parse_text(text: str) -> ParseResult:
    text = _clean(text)
    meta = {"format": "paste", **structural_signals(text)}
    meta.update(parse_confidence(meta))
    return ParseResult(text=text, source="paste", meta=meta)


def parse_confidence(meta: dict) -> dict[str, Any]:
    """How far to trust everything computed from this parse.

    Stage 1 required that a poor parse warn the user rather than yield a
    confident but wrong analysis (C2). The score blends five observable
    signals; below LOW_CONFIDENCE the interface shows a warning above the
    results. It estimates extraction quality, not resume quality.
    """
    notes: list[str] = []
    pages = max(meta.get("pages") or 1, 1)
    density = min(meta.get("char_count", 0) / (pages * 1500), 1.0)
    sections = min(len(meta.get("sections_found") or []) / 4, 1.0)
    contact = 0.5 * bool(meta.get("has_email")) + 0.5 * bool(meta.get("has_phone"))
    layout = 1.0
    if meta.get("multi_column"):
        layout -= 0.5
        notes.append("Multi-column layout: reading order may be scrambled.")
    if meta.get("layout_tables"):
        layout -= 0.25
        notes.append("Layout tables: cells may be merged into run-on lines.")
    if meta.get("textbox_count"):
        layout -= 0.25
        notes.append("Text boxes: their content may be skipped.")
    if meta.get("contact_in_header"):
        layout -= 0.25
        notes.append("Contact details sit in the page header.")
    layout = max(layout, 0.0)
    content = min(meta.get("bullet_count", 0) / 4, 1.0)
    if sections < 0.75:
        notes.append("Few standard section headings were recognised.")
    if not contact:
        notes.append("No email or phone number found in the body.")
    if density < 0.3:
        notes.append("Very little text was extracted for the page count.")

    score = round(0.25 * density + 0.25 * sections + 0.2 * contact + 0.2 * layout + 0.1 * content, 2)
    return {
        "parse_confidence": score,
        "parse_confidence_level": "low" if score < LOW_CONFIDENCE else "ok",
        "parse_confidence_notes": notes,
    }


# --------------------------------------------------------------------------
# ATS checks derived from the signals above
# --------------------------------------------------------------------------

def ats_checks(meta: dict) -> list[dict]:
    checks: list[dict] = []

    def add(check: str, status: str, note: str, group: str = "parse") -> None:
        # group "parse": can a machine read the file? "readability": does the
        # content read well once parsed? Kept apart because they have different
        # fixes and must not be blended (Stage 1, section 3.5).
        checks.append({"check": check, "status": status, "note": note, "group": group})

    if meta.get("format") == "pdf":
        fonts = meta.get("unembedded_fonts") or []
        if fonts:
            add("Fonts not embedded", "warn",
                f"{', '.join(fonts[:3])} {'is' if len(fonts) == 1 else 'are'} not embedded in the file. "
                "A parser may substitute them and garble the text. Re-export with fonts embedded.")
        if meta.get("multi_column"):
            add("Multi-column layout", "fail",
                "Text reads in a different order depending on how the parser walks the page. "
                "Columns scramble on most ATS. Move to a single column.")
        else:
            add("Single-column layout", "pass", "Reading order is stable — an ATS will get your lines in order.")
    if meta.get("layout_tables"):
        add("Tables in the layout", "fail",
            f"{meta.get('table_count')} table(s) found. Many parsers flatten a table row into one "
            "run-on line, merging unrelated fields.")
    if meta.get("textbox_count"):
        add("Text boxes", "fail",
            "Content inside text boxes is skipped entirely by common parsers.")
    if meta.get("contact_in_header"):
        add("Contact details in the page header", "fail",
            "Your email sits in the document header, which most parsers never read. Move it into the body.")

    if meta.get("has_email") and meta.get("has_phone"):
        add("Contact block is machine-readable", "pass", "Email and phone are in plain body text.")
    elif not meta.get("has_email"):
        add("No email found in the body", "fail", "Without a parseable email the application can dead-end.")

    sections = meta.get("sections_found") or []
    if len(sections) >= 3:
        add("Standard section headings", "pass", f"Found: {', '.join(sections[:5])}.")
    else:
        add("Non-standard section headings", "warn",
            "Fewer than three recognisable headings. Use plain words like Education, Experience, Projects, Skills.")

    ratio = meta.get("quantified_ratio", 0)
    total = meta.get("bullet_count", 0)
    if total and ratio < 0.3:
        add("Few quantified outcomes", "fail",
            f"{meta.get('quantified_bullets')} of {total} bullets contain a number. "
            "Recruiters scan for numbers first.", "readability")
    elif total:
        add("Quantified outcomes", "pass", f"{int(ratio * 100)}% of bullets carry a number.", "readability")

    weak = meta.get("weak_verb_bullets") or []
    if weak:
        add(f"Weak opening verbs on {len(weak)} bullet(s)", "fail",
            "Verbs like 'helped', 'worked on' and 'handled' describe presence, not contribution.",
            "readability")

    if not meta.get("has_code_link"):
        add("No link to code", "warn", "No GitHub or portfolio link found.", "readability")
    else:
        add("Code link present", "pass", "A reviewer can see your work without asking.", "readability")

    return checks
