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
    "goal_and_planning": [
        "goal",
        "objective",
        "target",
        "plan",
        "planning",
        "strategy",
        "roadmap",
        "timeline",
        "milestone",
    ],
    "workflow_and_execution": [
        "workflow",
        "process",
        "steps",
        "checklist",
        "task",
        "project",
        "implementation",
        "build",
        "deploy",
    ],
    "problem_solving": [
        "problem",
        "issue",
        "error",
        "bug",
        "fix",
        "troubleshoot",
        "debug",
        "resolve",
        "solution",
    ],
    "decision_and_comparison": [
        "decide",
        "decision",
        "choose",
        "option",
        "comparison",
        "tradeoff",
        "recommendation",
        "evaluate",
        "review",
    ],
}

TEXT_CUE_PATTERNS = {
    "goal_intent": re.compile(
        r"\b(i need to|i want to|my goal is|objective is|aim to|trying to)\b",
        flags=re.IGNORECASE,
    ),
    "planning_language": re.compile(
        r"\b(plan|steps?|roadmap|timeline|strategy|prioritize|sequence)\b",
        flags=re.IGNORECASE,
    ),
    "decision_process": re.compile(
        r"\b(option|trade[- ]?off|decide|decision|choose between|pros and cons)\b",
        flags=re.IGNORECASE,
    ),
    "obstacle_signal": re.compile(
        r"\b(stuck|blocked|problem|issue|error|difficulty|challenge|constraint)\b",
        flags=re.IGNORECASE,
    ),
    "monitoring_signal": re.compile(
        r"\b(progress|track|milestone|status|measure|benchmark)\b",
        flags=re.IGNORECASE,
    ),
    "persistence_signal": re.compile(
        r"\b(try again|retry|keep going|despite|still|even though|adapt|iterate|persevere)\b",
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
    "goal_specificity",
    "goal_commitment",
    "planning_specificity",
    "problem_decomposition",
    "sequencing_prioritization",
    "progress_monitoring_orientation",
    "obstacle_diagnosis_quality",
    "adaptation_strategy_quality",
    "help_seeking_effectiveness",
    "resource_tool_leverage",
    "execution_readiness",
    "self_efficacy_signal",
    "persistence_recovery",
    "reflective_learning_orientation",
    "agency_abdication",
    "fatalism_helplessness",
]

POSITIVE_ATTRIBUTES = AGENCY_ATTRIBUTES[:14]
NEGATIVE_ATTRIBUTES = AGENCY_ATTRIBUTES[14:]


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
