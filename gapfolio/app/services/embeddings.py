"""Embeddings.

One interface, three providers. `hash` is a deterministic offline stub so the
whole pipeline — including tests and a laptop demo — runs with no API key and
no network. It is a hashed bag-of-bigrams projection: genuinely similar strings
score higher than unrelated ones, but it is *not* semantic. Ship with voyage or
openai; the stub exists so nothing blocks on a key.
"""
from __future__ import annotations

import hashlib
import logging
import re
import threading
from typing import Iterable

import numpy as np

from ..config import settings

log = logging.getLogger("gapfolio.embeddings")

_TOKEN_RE = re.compile(r"[a-z0-9+#.]+")


def _tokens(text: str) -> list[str]:
    words = _TOKEN_RE.findall(text.lower())
    grams = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
    return grams


def _hash_embed(text: str, dim: int) -> list[float]:
    vec = np.zeros(dim, dtype=np.float32)
    for tok in _tokens(text):
        h = hashlib.blake2b(tok.encode(), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "big") % dim
        sign = 1.0 if h[4] % 2 else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    return (vec / norm).tolist() if norm else vec.tolist()


def _remote_embed(texts: list[str]) -> list[list[float]]:
    import httpx

    if settings.embedding_provider == "voyage":
        url = "https://api.voyageai.com/v1/embeddings"
        payload = {"input": texts, "model": settings.embedding_model, "input_type": "document"}
    else:  # openai-compatible
        url = "https://api.openai.com/v1/embeddings"
        payload = {"input": texts, "model": settings.embedding_model}

    headers = {"Authorization": f"Bearer {settings.embedding_api_key}"}
    with httpx.Client(timeout=45) as client:
        r = client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()["data"]
    return [row["embedding"] for row in sorted(data, key=lambda d: d.get("index", 0))]


_sbert = None
_sbert_failed = False
_sbert_lock = threading.Lock()


def _load_sbert():
    """Load the local model once. A failed load is remembered so every request
    does not retry a multi-second import; the service degrades to the stub and
    /api/health says so."""
    global _sbert, _sbert_failed
    if _sbert is not None or _sbert_failed:
        return _sbert
    with _sbert_lock:
        if _sbert is None and not _sbert_failed:
            try:
                from sentence_transformers import SentenceTransformer

                _sbert = SentenceTransformer(settings.sbert_model)
            except Exception as exc:  # missing package, no model cache, no network
                _sbert_failed = True
                log.warning("SBERT unavailable (%s); falling back to hash embeddings", exc)
    return _sbert


def effective_provider() -> str:
    """What `embed` will actually use right now."""
    if settings.embedding_provider == "sbert":
        return "sbert" if _load_sbert() is not None else "hash (sbert unavailable)"
    if settings.embedding_provider in ("voyage", "openai") and settings.embedding_api_key:
        return settings.embedding_provider
    return "hash"


def embed(texts: Iterable[str]) -> list[list[float]]:
    items = [t.strip() for t in texts if t and t.strip()]
    if not items:
        return []
    provider = settings.embedding_provider
    if provider == "sbert":
        model = _load_sbert()
        if model is not None:
            vecs = model.encode(items, normalize_embeddings=True, batch_size=64,
                                show_progress_bar=False)
            return [v.tolist() for v in vecs]
    elif provider in ("voyage", "openai") and settings.embedding_api_key:
        return [_unit(v) for v in _remote_embed(items)]
    return [_hash_embed(t, settings.embedding_dim) for t in items]


def embed_one(text: str) -> list[float]:
    out = embed([text])
    return out[0] if out else [0.0] * settings.embedding_dim


def _unit(v: list[float]) -> list[float]:
    arr = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(arr)
    return (arr / n).tolist() if n else arr.tolist()


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    x, y = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denom) if denom else 0.0


# --------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------

_SPLIT_RE = re.compile(r"^\s*(?:[-•▪◦*·–—]|\d+[.)])\s+")


def chunk_resume(text: str) -> list[tuple[str, str]]:
    """Return (chunk, kind) pairs. Bullets and skill-list entries are indexed
    separately so a skill named in a list can be told apart from one that has a
    project behind it — that distinction is what 'partial' status depends on."""
    from .entities import segment_sections

    chunks: list[tuple[str, str]] = []
    # Segmenting first means a skills section is recognised even under a
    # non-standard heading ("Toolbox"), so its entries are still tagged as
    # listed rather than demonstrated.
    for seg in segment_sections(text):
        in_skills = seg.section == "skills"
        if seg.heading and not in_skills:
            chunks.append((seg.heading, "section"))
        for line in seg.lines:
            if in_skills and ("," in line or "|" in line):
                for item in re.split(r"[,|/]", line):
                    item = item.strip()
                    if 1 < len(item) < 60:
                        chunks.append((item, "skill_listed"))
                continue
            if _SPLIT_RE.match(line):
                chunks.append((_SPLIT_RE.sub("", line), "achievement"))
            elif len(line) > 25:
                chunks.append((line, "context"))
    return chunks[:120]


def chunk_jd(text: str) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or len(line) < 12:
            continue
        kind = "requirement" if _SPLIT_RE.match(line) else "context"
        chunks.append((_SPLIT_RE.sub("", line), kind))
    return chunks[:80]
