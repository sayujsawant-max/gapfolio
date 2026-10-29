"""LLM access.

Used for the two things embeddings can't do: pulling structured requirements
out of a job posting, and writing the fix for each gap. Scoring itself is *not*
delegated to the model — that runs on embeddings plus outcome-calibrated
weights, so it is reproducible and auditable.
"""
from __future__ import annotations

import json
import re
from typing import Any

import httpx

from ..config import settings

API_URL = "https://api.anthropic.com/v1/messages"


class LLMUnavailable(RuntimeError):
    pass


def available() -> bool:
    return bool(settings.anthropic_api_key)


def complete_json(prompt: str, max_tokens: int = 2000) -> dict[str, Any]:
    if not available():
        raise LLMUnavailable("No Anthropic API key configured.")
    headers = {
        "x-api-key": settings.anthropic_api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": settings.anthropic_model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    with httpx.Client(timeout=90) as client:
        r = client.post(API_URL, json=body, headers=headers)
        if r.status_code >= 400:
            raise LLMUnavailable(f"Model API returned {r.status_code}")
        data = r.json()
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    return loads(text)


def loads(text: str) -> dict[str, Any]:
    """Parse model output that should be JSON, tolerating fences and truncation."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start = cleaned.find("{")
    if start > 0:
        cleaned = cleaned[start:]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    depth, in_str, esc, last_comma = 0, False, False, -1
    for i, ch in enumerate(cleaned):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        elif ch == "," and depth <= 2:
            last_comma = i
    if last_comma > 0:
        salvaged = cleaned[:last_comma]
        opens = len(re.findall(r"[\[{]", salvaged))
        closes = len(re.findall(r"[\]}]", salvaged))
        for _ in range(opens - closes):
            salvaged += "]" if salvaged.rfind("[") > salvaged.rfind("{") else "}"
        return json.loads(salvaged)
    raise LLMUnavailable("Model response was not valid JSON.")


REQUIREMENTS_PROMPT = """Extract the hiring requirements from this job description.

JOB DESCRIPTION:
\"\"\"{jd}\"\"\"

Return 8-12 requirements. For each:
- label: the skill or capability in 2-6 words
- jd_quote: copied character-for-character from the text above, 5-14 words, so it can be located in the original
- weight: "core" if the role fails without it, "nice-to-have" otherwise
- family: one of backend, frontend, data, devops, mobile, design, product, general

Also return role_title, company (or ""), seniority, and role_family.

Return ONLY minified JSON:
{{"role_title":"","company":"","seniority":"","role_family":"","requirements":[{{"label":"","jd_quote":"","weight":"core","family":""}}]}}"""


ROADMAP_PROMPT = """A candidate targeting "{role}" has these gaps, already ranked by impact and effort.
Each line shows the gap, its status, and the only resume text that bears on it.

{gap_lines}

WEAK BULLETS (verbatim from the resume, flagged by a rule-based check):
{weak_lines}

You have been given structured findings, not the resume. Do not assume anything about the
candidate beyond what is written above. Never invent experience, employers, tools or numbers.

Write:
- roadmap: one entry per gap above, in the same order. fix_type is "course", "project" or "rewrite". action is one specific sentence naming the actual thing to build or take. proof is the resume bullet they would be able to write afterwards; write every number as the placeholder [N], never a real figure.
- rewrites: up to 3, each "before" copied character-for-character from the WEAK BULLETS above. "after" may only reuse numbers already present in "before"; use [N] where a measurement belongs.
- questions: 3 interview questions the gaps above invite for this role.

Be blunt and specific. No praise, no filler.

Return ONLY minified JSON:
{{"roadmap":[{{"skill":"","fix_type":"","action":"","proof":""}}],"rewrites":[{{"before":"","after":"","why":""}}],"questions":["","",""]}}"""
