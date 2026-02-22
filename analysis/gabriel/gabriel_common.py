from __future__ import annotations

import argparse
import html
import json
import os
import re
from pathlib import Path
from typing import Any

TEXT_JOIN_DELIMITER = "\n\n---\n\n"
DEFAULT_MIN_SNIPPET_CHARS = 300
MAX_SNIPPET_CHARS = 700

TITLE_KEYWORD_GROUPS = {
    "personal_agency": [
        "goal",
        "objective",
        "plan",
        "strategy",
        "roadmap",
        "timeline",
        "milestone",
        "execute",
        "implementation",
        "ownership",
        "self-directed",
        "self-management",
        "action plan",
    ],
    "proxy_agency": [
        "assistant",
        "advisor",
        "coach",
        "agent",
        "chatbot",
        "ai",
        "ai assistant",
        "copilot",
        "recommendation",
        "guidance",
        "delegate",
        "automate",
        "decide for me",
        "tell me what to do",
    ],
    "collective_agency": [
        "team",
        "together",
        "collaborate",
        "coordination",
        "co-worker",
        "shared",
        "cross-functional",
        "stakeholder",
        "joint",
        "handoff",
        "align",
    ],
}

TEXT_CUE_PATTERNS = {
    "personal_intent_control": re.compile(
        r"\b(i (will|can|plan to|intend to|am going to)|my plan is|i'll)\b",
        flags=re.IGNORECASE,
    ),
    "personal_ownership_responsibility": re.compile(
        r"\b(i (take|own|accept) responsibility|my responsibility|on my own|i(?:'ll| will) handle)\b",
        flags=re.IGNORECASE,
    ),
    "personal_self_efficacy": re.compile(
        r"\b(i(?:'m| am) confident|i can handle|i can do this|i know how to)\b",
        flags=re.IGNORECASE,
    ),
    "proxy_advice_seeking": re.compile(
        r"\b(can you|could you|please (recommend|advise|choose|decide)|what do you recommend|tell me what to do)\b",
        flags=re.IGNORECASE,
    ),
    "proxy_delegation": re.compile(
        r"\b(decide for me|you choose|you decide|delegate (this|it)|hand this off|automate this for me)\b",
        flags=re.IGNORECASE,
    ),
    "proxy_trust_dependence": re.compile(
        r"\b(i (trust|rely on|depend on) (you|the assistant|this chatbot|the ai)|follow your advice)\b",
        flags=re.IGNORECASE,
    ),
    "collective_team_coordination": re.compile(
        r"\b(our team|as a team|work together|collaborat(e|ion)|coordinate with|sync with|align with)\b",
        flags=re.IGNORECASE,
    ),
    "collective_role_split": re.compile(
        r"\b(while you|while the team|loop in|handoff to|hand off to|escalate to)\b",
        flags=re.IGNORECASE,
    ),
    "collective_shared_efficacy": re.compile(
        r"\b(shared goal|joint effort|jointly|cross-functional|co-own|coordinated effort)\b",
        flags=re.IGNORECASE,
    ),
}

EMAIL_PATTERN = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", flags=re.IGNORECASE
)
URL_PATTERN = re.compile(r"\b(?:https?://|www\.)\S+\b", flags=re.IGNORECASE)
PHONE_PATTERN = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?!\w)"
)
LONG_ID_PATTERN = re.compile(
    r"\b[a-f0-9]{16,}\b|\b[A-Za-z0-9_-]{20,}\b", flags=re.IGNORECASE
)

SENTENCE_BREAKS = [". ", "? ", "! ", "; "]

AGENCY_ATTRIBUTES = [
    "personal_agency",
    "proxy_agency",
    "collective_agency",
]

AGENCY_ATTRIBUTE_DEFINITIONS = {
    "personal_agency": (
        "Personal (self) agency: the user's own capability and perceived control to pursue "
        "their goals through their own actions. Score higher when the text shows clear "
        "ownership, ability, and intentional action to produce outcomes."
    ),
    "proxy_agency": (
        "Proxy agency: reliance on another agent's capability (for example, an AI assistant) "
        "to help the user achieve goals when the user lacks enough capability or control alone. "
        "Score higher when the user expresses trust in the proxy and transfer of decision/action control."
    ),
    "collective_agency": (
        "Collective agency: coordinated capability and control across a team of actors "
        "(user + AI + human collaborators) working together toward a shared service goal. "
        "Score higher when the text reflects teamwork, coordination, and shared efficacy."
    ),
}


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    default_input: str,
    default_output: str,
) -> argparse.ArgumentParser:
    parser.add_argument("--input", default=default_input, help="Input dataset path.")
    parser.add_argument("--output", default=default_output, help="Primary output path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--model-id",
        default="gpt-5-mini",
        help="Model identifier (used by Gabriel scripts).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional record limit for smoke runs.",
    )
    return parser


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def clean_title(title: str | None) -> str:
    if title is None:
        return "Untitled conversation"
    cleaned = html.unescape(title)
    cleaned = re.sub(r"\bx27\b", "'", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("\\n", " ")
    cleaned = normalize_whitespace(cleaned)
    return cleaned or "Untitled conversation"


def redact_text(text: str) -> str:
    redacted = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)
    redacted = URL_PATTERN.sub("[REDACTED_URL]", redacted)
    redacted = PHONE_PATTERN.sub("[REDACTED_PHONE]", redacted)
    redacted = LONG_ID_PATTERN.sub("[REDACTED_ID]", redacted)
    return redacted


def score_title(title: str) -> tuple[int, list[str]]:
    title_text = title.lower()
    matched_terms: set[str] = set()
    for terms in TITLE_KEYWORD_GROUPS.values():
        for term in terms:
            if re.search(rf"\b{re.escape(term)}\b", title_text):
                matched_terms.add(term)
    return len(matched_terms), sorted(matched_terms)


def score_text_cues(text: str) -> tuple[int, list[str]]:
    matched_cues = [cue for cue, pattern in TEXT_CUE_PATTERNS.items() if pattern.search(text)]
    return len(matched_cues), matched_cues


def load_env_file(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def read_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_zscore(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = var**0.5
    if std == 0:
        return [0.0 for _ in values]
    return [(v - mean) / std for v in values]


def _split_long_segment(segment: str, max_chars: int) -> list[str]:
    text = normalize_whitespace(segment)
    if len(text) <= max_chars:
        return [text] if text else []

    pieces: list[str] = []
    rest = text
    while len(rest) > max_chars:
        candidate = rest[:max_chars]
        boundary = -1
        for mark in SENTENCE_BREAKS:
            boundary = max(boundary, candidate.rfind(mark))
        if boundary >= int(max_chars * 0.6):
            piece = candidate[: boundary + 1].strip()
        else:
            piece = candidate.strip()
        pieces.append(piece)
        rest = rest[len(piece) :].strip()
    if rest:
        pieces.append(rest)
    return [p for p in pieces if p]


def build_deterministic_chunks(
    text: str,
    *,
    max_chars_per_chunk: int,
    max_chunks: int,
) -> tuple[list[str], int, float]:
    raw = text or ""
    total_text = normalize_whitespace(raw)
    if not total_text:
        return [], 0, 0.0

    raw_segments = [normalize_whitespace(seg) for seg in raw.split(TEXT_JOIN_DELIMITER)]
    segments = [seg for seg in raw_segments if seg]
    if not segments:
        segments = [total_text]

    expanded_segments: list[str] = []
    for seg in segments:
        expanded_segments.extend(_split_long_segment(seg, max_chars=max_chars_per_chunk))

    all_chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    joiner = "\n\n---\n\n"
    joiner_len = len(joiner)
    for seg in expanded_segments:
        seg_len = len(seg)
        if not current:
            current = [seg]
            current_len = seg_len
            continue

        projected_len = current_len + joiner_len + seg_len
        if projected_len <= max_chars_per_chunk:
            current.append(seg)
            current_len = projected_len
        else:
            all_chunks.append(joiner.join(current))
            current = [seg]
            current_len = seg_len
    if current:
        all_chunks.append(joiner.join(current))

    n_chunks_total = len(all_chunks)
    selected = all_chunks[:max_chunks]
    if not selected:
        return [], n_chunks_total, 0.0

    included_chars = sum(len(chunk) for chunk in selected)
    full_chars = sum(len(chunk) for chunk in all_chunks)
    coverage = included_chars / max(full_chars, 1)
    return selected, n_chunks_total, coverage
