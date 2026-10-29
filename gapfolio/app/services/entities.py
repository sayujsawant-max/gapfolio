"""Section segmentation and skill-entity extraction.

Stage 1 specified a heading rulebook with a classification fallback for
non-standard documents, and an entity extractor built on a custom skill ruler.

Segmentation runs in two tiers. A rulebook of synonyms recognises the headings
real resumes use ("Work Experience", "Technical Skills"). A heading it does not
recognise ("Things I Made", "Toolbox") is classified by embedding similarity
against a short descriptor per section, using the heading plus the first lines
beneath it as context.

Standard and inferred sections are kept apart on purpose. An ATS only
recognises the standard headings, so the ATS audit counts those. The matcher
cares what the content *is*, so it uses both.

Skill extraction is a phrase ruler over a curated vocabulary, the same design
as a spaCy EntityRuler but with no model dependency. It finds skills the
vocabulary knows. It cannot find ones it does not, and the evaluation reports
that limit rather than hiding it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------

SECTION_SYNONYMS: dict[str, set[str]] = {
    "education": {"education", "academic background", "academics", "qualifications",
                  "academic qualifications", "educational qualifications"},
    "experience": {"experience", "work experience", "professional experience",
                   "employment", "employment history", "internship", "internships",
                   "work history", "industrial experience"},
    "projects": {"projects", "academic projects", "personal projects", "key projects",
                 "selected projects", "project work"},
    "skills": {"skills", "technical skills", "core skills", "key skills",
               "core competencies", "technologies", "tech stack", "skills and tools",
               "skills & tools"},
    "certifications": {"certifications", "certificates", "courses", "licenses",
                       "licences", "training"},
    "achievements": {"achievements", "awards", "honours", "honors", "accomplishments",
                     "awards and achievements", "awards & achievements"},
    "summary": {"summary", "objective", "profile", "career objective",
                "professional summary", "about me"},
    "activities": {"activities", "extracurricular", "extracurricular activities",
                   "positions of responsibility", "positions", "leadership",
                   "volunteering"},
    "publications": {"publications", "research", "papers"},
}
_HEADING_TO_SECTION = {h: s for s, hs in SECTION_SYNONYMS.items() for h in hs}

# What each section's content looks like, for classifying an unknown heading.
# Several short descriptors per section, scored by best match: a single
# keyword string makes similarity weak and the sections hard to tell apart.
SECTION_DESCRIPTORS: dict[str, list[str]] = {
    "education": ["education", "degree, university, college or school", "B.Tech, CGPA and graduation year"],
    "experience": ["work experience", "internship at a company", "intern or employee role at an employer",
                   "Backend Intern at a startup, May to July 2026",
                   "worked as an intern, fixed bugs and shipped features for the team"],
    "projects": ["projects", "built and developed an application or system", "personal and academic projects",
                 "Campus app project: built a web application with a REST API and a database",
                 "trained and evaluated a model, deployed a working demo"],
    "skills": ["skills", "technical skills, programming languages and tools", "Python, Java, SQL, Git, Docker"],
    "certifications": ["certifications", "completed an online course and earned a certificate"],
    "achievements": ["achievements and awards", "won a competition, rank or recognition"],
    "summary": ["professional summary", "career objective and profile", "motivated student seeking a role"],
    "activities": ["extracurricular activities", "clubs, leadership and volunteering"],
}
FALLBACK_MIN_SIM = 0.32
_descriptor_cache: dict[str, list[tuple[str, list[float]]]] = {}

_BULLET = re.compile(r"^\s*(?:[-•▪◦*·–—]|\d+[.)])\s+")


@dataclass
class Segment:
    section: str                 # canonical section name, or "header"
    heading: str                 # the line as written
    lines: list[str] = field(default_factory=list)
    inferred: bool = False       # True when the heading was classified, not recognised


def _is_heading_shape(line: str) -> bool:
    s = line.strip().rstrip(":")
    if not s or len(s) > 40 or _BULLET.match(line) or "@" in s or "," in s:
        return False
    words = s.split()
    if not 1 <= len(words) <= 5 or s.endswith((".", ";")):
        return False
    if sum(c.isdigit() for c in s) > 2:
        return False
    # str.istitle() rejects "What I've Made"; test the first letter of each word.
    capitalised = all(w[0].isupper() for w in words if w[0].isalpha())
    return s.isupper() or capitalised or line.strip().endswith(":") or s.lower() in _HEADING_TO_SECTION


def _descriptors() -> list[tuple[str, list[float]]]:
    """(section, vector) for every descriptor, embedded once per provider."""
    from .embeddings import embed, effective_provider

    key = effective_provider()
    if key not in _descriptor_cache:
        pairs = [(s, d) for s, ds in SECTION_DESCRIPTORS.items() for d in ds]
        vecs = embed([d for _, d in pairs])
        _descriptor_cache[key] = [(s, v) for (s, _), v in zip(pairs, vecs)]
    return _descriptor_cache[key]


SWITCH_MARGIN = 0.05


def _classify_heading(heading: str, context: list[str], current: str = "") -> str | None:
    """The section whose descriptors sit closest to the heading plus the lines
    beneath it, or None if nothing is close enough. The lines matter: "My
    Schooling" alone is ambiguous, "My Schooling / B.Tech CSE, 2027" is not.
    Imported lazily so parsing has no hard model dependency."""
    from .embeddings import cosine, embed

    descriptors = _descriptors()
    qvec = embed([" ".join([heading] + context[:3])])
    if not qvec:
        return None
    best: dict[str, float] = {}
    for section, vec in descriptors:
        best[section] = max(best.get(section, -1.0), cosine(qvec[0], vec))
    for section, bonus in _content_cues(context[:3]).items():
        best[section] = best.get(section, 0.0) + bonus
    section = max(best, key=best.get)
    if best[section] < FALLBACK_MIN_SIM:
        return None
    # Inertia: a line that merely resembles another section as much as the one
    # already open (a project title in Projects) does not start a new section.
    if current in best and section != current and best[section] - best[current] < SWITCH_MARGIN:
        return None
    return section


_DEGREE = re.compile(r"\b(b\.?\s?tech|b\.?e\b|m\.?\s?tech|bachelor|master|cgpa|gpa|university|college|institute|hsc|ssc)\b", re.I)
_ROLE = re.compile(r"\b(intern|internship|engineer|developer|analyst|trainee|associate|consultant)\b", re.I)
_YEAR = re.compile(r"\b(?:19|20)\d\d\b")
_ACTION = re.compile(
    r"^\s*(?:[-•▪◦*·–—]\s*)?(built|developed|designed|implemented|created|trained|deployed|wrote|"
    r"redesigned|fine-tuned|set up|configured|ran|launched)\b", re.I)
_CERT = re.compile(r"\b(certif\w*|coursera|udemy|nptel|course)\b", re.I)


def _content_cues(context: list[str]) -> dict[str, float]:
    """Structural cues in the lines under a heading. Embedding similarity alone
    is weak for short, abstract headings ("Industry Exposure"); the lines below
    are usually unmistakable: a degree and CGPA, a role with a date, a comma
    separated list. Each cue adds a fixed bonus to its section's score."""
    cues: dict[str, float] = {}
    joined = " ".join(context)
    if _DEGREE.search(joined):
        cues["education"] = 0.20
    if _ROLE.search(joined) and _YEAR.search(joined):
        cues["experience"] = 0.15
    if context and context[0].count(",") >= 2 and all(len(p.split()) <= 3 for p in context[0].split(",")):
        cues["skills"] = 0.20
    if any(_ACTION.match(l) for l in context):
        cues["projects"] = cues.get("projects", 0.0) + 0.08
    if _CERT.search(joined):
        cues["certifications"] = 0.15
    return cues


def segment_sections(text: str, fallback: bool = True) -> list[Segment]:
    lines = [l.rstrip() for l in text.splitlines()]
    segments: list[Segment] = [Segment("header", "")]
    seen = 0  # non-empty lines so far; the first two are the name and contact line
    for idx, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        seen += 1
        key = line.rstrip(":").strip().lower()
        if key in _HEADING_TO_SECTION:
            segments.append(Segment(_HEADING_TO_SECTION[key], line))
            continue
        if fallback and seen > 2 and _is_heading_shape(raw):
            # A heading-shaped line (short, capitalised, no sentence punctuation)
            # that the rulebook does not know. Blank lines are not required:
            # DOCX parsing drops them. What stops a project title ("Campus
            # Payments Service") opening a new section is that it classifies as
            # the section already open, and anything below the similarity
            # threshold is left as an ordinary line.
            context = []
            for nxt in lines[idx + 1: idx + 5]:
                nxt = nxt.strip()
                if not nxt:
                    continue
                if nxt.rstrip(":").lower() in _HEADING_TO_SECTION:
                    break  # the next section's lines say nothing about this one
                if context and _is_heading_shape(nxt):
                    break  # nor do the lines under an unrecognised heading
                context.append(nxt)
            lone_skill = len(line.split()) <= 2 and bool(extract_skills(line))
            section = (None if lone_skill or not context
                       else _classify_heading(line.rstrip(":"), context, segments[-1].section))
            if section and section != segments[-1].section:
                segments.append(Segment(section, line, inferred=True))
                continue
        segments[-1].lines.append(line)
    return [s for s in segments if s.heading or s.lines]


def standard_sections(text: str) -> list[str]:
    """Sections whose heading an ATS would recognise (no fallback involved)."""
    return sorted({s.section for s in segment_sections(text, fallback=False) if s.heading})


# --------------------------------------------------------------------------
# skills
# --------------------------------------------------------------------------

# canonical name -> alternative spellings. Matching is case-insensitive on word
# boundaries; "c" and "r" are deliberately absent because they match everywhere.
SKILLS: dict[str, list[str]] = {
    "python": [], "java": [], "javascript": ["js", "ecmascript"], "typescript": ["ts"],
    "c++": ["cpp"], "c#": ["csharp"], "golang": [], "rust": [], "kotlin": [], "swift": [],
    "php": [], "ruby": [], "scala": [], "matlab": [], "bash": ["shell scripting"],
    "html": ["html5"], "css": ["css3"], "sql": [], "nosql": [],
    "react": ["react.js", "reactjs"], "angular": ["angularjs"], "vue": ["vue.js", "vuejs"],
    "next.js": ["nextjs"], "redux": [], "tailwind": ["tailwind css"], "bootstrap": [],
    "node.js": ["nodejs", "node"], "express": ["express.js", "expressjs"],
    "django": [], "flask": [], "fastapi": [], "spring boot": ["springboot"], "spring": [],
    ".net": ["dotnet", "asp.net"], "graphql": [], "rest api": ["rest apis", "restful api", "restful apis"],
    "mysql": [], "postgresql": ["postgres"], "sqlite": [], "mongodb": ["mongo"],
    "redis": [], "oracle": [], "firebase": [], "elasticsearch": [],
    "docker": [], "kubernetes": ["k8s"], "terraform": [], "ansible": [],
    "jenkins": [], "github actions": [], "gitlab ci": [], "ci/cd": ["cicd", "ci cd"],
    "git": [], "github": [], "gitlab": [], "linux": [], "nginx": [],
    "aws": ["amazon web services"], "azure": [], "gcp": ["google cloud"],
    "ec2": [], "s3": [], "lambda": [],
    "pandas": [], "numpy": [], "scikit-learn": ["sklearn", "scikit learn"],
    "tensorflow": [], "pytorch": [], "keras": [], "opencv": [], "nltk": [], "spacy": [],
    "machine learning": ["ml"], "deep learning": [], "nlp": ["natural language processing"],
    "computer vision": [], "data analysis": [], "statistics": [],
    "tableau": [], "power bi": ["powerbi"], "excel": ["ms excel"], "matplotlib": [],
    "seaborn": [], "airflow": [], "spark": ["pyspark"], "hadoop": [], "kafka": [],
    "rabbitmq": [], "grafana": [], "prometheus": [],
    "pytest": [], "junit": [], "jest": [], "selenium": [], "postman": [],
    "unit testing": ["unit tests"], "figma": [], "jira": [],
    "data structures": ["dsa"], "algorithms": [], "oop": ["object oriented programming"],
    "agile": ["scrum"], "android": [], "flutter": [], "react native": [],
}

_ALIAS_TO_CANON: dict[str, str] = {}
for _canon, _alts in SKILLS.items():
    _ALIAS_TO_CANON[_canon] = _canon
    for _a in _alts:
        _ALIAS_TO_CANON[_a] = _canon

# Longest phrases first so "spring boot" wins over "spring" and "react native"
# over "react".
_PHRASES = sorted(_ALIAS_TO_CANON, key=len, reverse=True)
_SKILL_RE = re.compile(
    r"(?<![\w+#.])(" + "|".join(re.escape(p) for p in _PHRASES) + r")(?![\w+#]|\.\w)",
    re.IGNORECASE,
)


# Skill names that are also ordinary English words. They only count when
# written capitalised ("Express", "Spring"), not mid-sentence ("express interest").
_AMBIGUOUS = {"express", "spring", "oracle", "swift", "ruby", "spark", "rust",
              "lambda", "agile", "node.js", "statistics", "algorithms"}


def extract_skills(text: str) -> list[str]:
    """Canonical skill names found in the text, in order of first appearance."""
    found: dict[str, None] = {}
    for m in _SKILL_RE.finditer(text):
        surface = m.group(1)
        canon = _ALIAS_TO_CANON[surface.lower()]
        if canon in _AMBIGUOUS and surface[0].islower() and surface.lower() == surface:
            # lower-case ambiguous word: accept only inside a list-like context
            start = max(0, m.start() - 2)
            if not re.search(r"[,|/]\s*$", text[start:m.start()]) and not re.match(r"\s*[,|/]", text[m.end():m.end() + 2]):
                continue
        found.setdefault(canon, None)
    return list(found)
