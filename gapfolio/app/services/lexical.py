"""Lexical overlap, the other half of hybrid retrieval.

Dense embeddings catch "client engagement" ≈ "customer relationship
management". They are weaker on exact technical tokens: "Docker" and
"Kubernetes" sit close together in embedding space precisely because they are
both container tooling, which is the opposite of what a requirement check
wants. Lexical overlap is unforgiving about that distinction.

Scoring one signal alone gets one of the two cases wrong, so matching blends
them. Prefix matching stands in for a stemmer, which keeps
containerise/containerised/containerisation together without a dependency.
"""
from __future__ import annotations

import re

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "for", "with", "to", "at",
    "by", "from", "as", "is", "are", "be", "you", "your", "we", "our", "will",
    "that", "this", "it", "its", "into", "using", "use", "used", "such", "including",
    "ability", "experience", "strong", "good", "excellent", "work", "working",
    "across", "within", "who", "can", "have", "has", "must", "should", "well",
    # Generic posting verbs and filler. These appear in nearly every
    # responsibility line, so they dilute the terms that actually distinguish
    # one requirement from another.
    "design", "build", "develop", "ensure", "deliver", "maintain", "support",
    "own", "join", "help", "role", "team", "day", "new", "also", "like",
    "etc", "plus", "bonus", "both", "other", "some", "them", "their",
}

_WORD = re.compile(r"[a-z0-9+#.]{2,}")
STEM_LEN = 6

# Compounds a naive tokeniser splits or mangles. "CI/CD" becoming "ci" + "cd"
# is why a GitHub Actions bullet failed to satisfy a CI/CD requirement.
_COMPOUNDS = [
    (re.compile(r"\bci\s*[/&-]\s*cd\b"), "cicd"),
    (re.compile(r"\bnode\.?js\b"), "nodejs"),
    (re.compile(r"\breact\.?js\b"), "react"),
    (re.compile(r"\bvue\.?js\b"), "vue"),
    (re.compile(r"\bk8s\b"), "kubernetes"),
    (re.compile(r"\bpostgres(?:ql)?\b"), "postgresql"),
    (re.compile(r"\bscikit[- ]learn\b"), "scikit"),
    (re.compile(r"\bgithub\s+actions\b"), "github actions"),
]
_SUFFIXES = ("ations", "ation", "ings", "ing", "ied", "ies", "ed", "es", "s")


def _normalise(text: str) -> str:
    text = text.lower()
    for pattern, repl in _COMPOUNDS:
        text = pattern.sub(repl, text)
    return text


def _stem(token: str) -> str:
    """Strip one common inflection, then truncate. handled/handling/handles all
    meet at "handl"; user/users at "user". Cheaper than a lemmatiser and
    deterministic, which matters because the stems are compared on both sides."""
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)][:STEM_LEN]
    return token[:STEM_LEN]


def tokens(text: str) -> set[str]:
    words = (w.strip(".") for w in _WORD.findall(_normalise(text)))
    return {w for w in words if len(w) >= 2 and w not in STOPWORDS}


def stems(text: str) -> set[str]:
    return {_stem(t) for t in tokens(text)}


# A named technology implies the category a posting usually asks for. Prefix
# stemming cannot connect "MySQL" to "PostgreSQL", and dense embeddings are
# unreliable here for the opposite reason — Docker and Kubernetes sit close
# together precisely because they are both container tooling. A small explicit
# ontology handles the cases that matter for fresher hiring in India, and it is
# auditable in a way a learned mapping is not.
#
# Expansion applies to the *evidence* side only. A resume saying MySQL should
# satisfy a requirement asking for SQL; a resume saying SQL should not be
# credited with PostgreSQL tuning it never mentions.
ALIASES: dict[str, set[str]] = {
    "mysql": {"sql", "rdbms", "database", "queries"},
    "postgresql": {"sql", "rdbms", "database", "postgres", "queries"},
    "postgres": {"sql", "rdbms", "database", "postgresql", "queries"},
    "sqlite": {"sql", "rdbms", "database"},
    "oracle": {"sql", "rdbms", "database"},
    "mongodb": {"database", "nosql"},
    "redis": {"cache", "caching", "database"},
    "sql": {"database", "rdbms", "queries"},
    "express": {"api", "apis", "rest", "backend", "nodejs", "node"},
    "fastapi": {"api", "apis", "rest", "backend"},
    "flask": {"api", "apis", "rest", "backend"},
    "django": {"api", "apis", "rest", "backend"},
    "spring": {"api", "apis", "rest", "backend"},
    "nodejs": {"node", "backend", "javascript"},
    "node": {"nodejs", "backend", "javascript"},
    "react": {"frontend", "javascript", "ui"},
    "angular": {"frontend", "javascript", "ui"},
    "vue": {"frontend", "javascript", "ui"},
    "docker": {"container", "containerise", "containerised", "containers"},
    "kubernetes": {"container", "containers", "orchestration", "k8s"},
    "jenkins": {"cicd", "pipelines", "pipeline", "deployment"},
    "github": {"git", "version", "cicd"},
    "gitlab": {"git", "version", "cicd"},
    "actions": {"cicd", "pipelines", "pipeline"},
    "login": {"authentication", "auth", "session"},
    "signup": {"authentication", "auth"},
    "jwt": {"authentication", "auth", "session"},
    "oauth": {"authentication", "auth"},
    "ec2": {"aws", "cloud", "deployment"},
    "s3": {"aws", "cloud", "storage"},
    "lambda": {"aws", "cloud", "serverless"},
    "aws": {"cloud"},
    "azure": {"cloud"},
    "gcp": {"cloud"},
    "pytest": {"tests", "testing", "unit"},
    "jest": {"tests", "testing", "unit"},
    "junit": {"tests", "testing", "unit"},
    "selenium": {"tests", "testing", "integration"},
    "pandas": {"data", "analysis", "python"},
    "numpy": {"data", "analysis", "python"},
    "scikit": {"machine", "learning", "model", "ml"},
    "tensorflow": {"machine", "learning", "model", "ml"},
    "pytorch": {"machine", "learning", "model", "ml"},
    "kafka": {"queue", "queues", "messaging", "streaming"},
    "rabbitmq": {"queue", "queues", "messaging"},
    "grafana": {"monitoring", "dashboards", "observability"},
    "prometheus": {"monitoring", "metrics", "observability"},
    "figma": {"design", "ui", "wireframes"},
    "c++": {"dsa", "algorithms", "programming"},
    "java": {"programming", "backend"},
    "python": {"programming"},
    "javascript": {"programming"},
}


def expand(text: str) -> set[str]:
    """Stems of the text plus the categories its named technologies imply."""
    raw = tokens(text)
    widened = set(raw)
    for t in raw:
        widened |= ALIASES.get(t, set())
    return {_stem(t) for t in widened}


def coverage(requirement: str, chunk: str, idf: dict[str, float] | None = None) -> float:
    """How much of what makes this requirement *distinctive* appears in the chunk.

    Asymmetric on purpose: a long achievement bullet containing every term of a
    short requirement scores 1.0 rather than being penalised for saying more.

    Plain term overlap fails on real postings, because a responsibility line
    reads "Design and build REST APIs in Node.js or Python that are reliable
    under load" — a dozen terms, of which only three carry the requirement.
    Scoring all of them equally buries the signal, so terms are IDF-weighted
    and only the most distinctive few count.
    """
    need = stems(requirement)
    if not need:
        return 0.0
    have = expand(chunk)
    if idf:
        # A term the resume never uses is the *most* distinctive thing the
        # requirement asks for, so unseen terms take the maximum weight. Giving
        # them a low default inverts the ranking and quietly hides real gaps.
        #
        # Every term stays in the denominator rather than only the top few:
        # keeping just the rarest terms makes the score effectively binary,
        # since a requirement almost always contains some term the resume has
        # never used, and one such term would zero out real partial evidence.
        ceiling = idf.get(_MAX_KEY, 1.0)
        weights = {t: idf.get(t, ceiling) for t in need}
        total = sum(weights.values())
        if not total:
            return 0.0
        return sum(w for t, w in weights.items() if t in have) / total
    return len(need & have) / len(need)




# build_idf stashes the ceiling weight under this key so `coverage` can score
# terms the resume never uses. It is not a real term, and stems() can never
# produce it, so it cannot collide with one.
_MAX_KEY = "\x00max"


def build_idf(corpus: list[str]) -> dict[str, float]:
    """IDF over the resume's own chunks.

    A term the resume uses everywhere ("build", "team") tells you nothing about
    whether a specific requirement is met; a term it uses once or never
    ("postgresql") tells you a great deal. Computed per resume, so it costs one
    pass and no external corpus.
    """
    import math

    n = len(corpus) or 1
    seen: dict[str, int] = {}
    for doc in corpus:
        for t in stems(doc):
            seen[t] = seen.get(t, 0) + 1
    idf = {t: math.log(1 + n / c) for t, c in seen.items()}
    # A term appearing in no chunk at all: the limit of the formula as c → 0,
    # clamped so one unseen term cannot swamp every seen one.
    idf[_MAX_KEY] = math.log(1 + n / 0.5)
    return idf
