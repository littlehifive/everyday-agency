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

CONSTRUCTS = [
    "goal_setting",
    "goal_navigation",
    "problem_decomposition",
    "strategic_planning",
    "progress_monitoring",
    "obstacle_management",
    "help_seeking_resourcefulness",
    "self_efficacy_or_confidence",
    "resilience_or_persistence",
]

CONSTRUCT_DEFINITIONS = {
    "goal_setting": "Explicit statement of a goal, objective, or desired outcome.",
    "goal_navigation": "Concrete movement toward a goal through sequencing, prioritizing, or adjusting steps.",
    "problem_decomposition": "Breaking a larger challenge into smaller tasks, causes, or sub-problems.",
    "strategic_planning": "Choosing a method, plan, or strategy before acting.",
    "progress_monitoring": "Checking progress, milestones, status, or performance against expectations.",
    "obstacle_management": "Identifying and working through blockers, errors, or constraints.",
    "help_seeking_resourcefulness": "Requesting tools, advice, references, or support to move forward.",
    "self_efficacy_or_confidence": "Language signaling confidence in ability to execute or learn.",
    "resilience_or_persistence": "Continuing effort despite setbacks; retrying, adapting, or persevering.",
}

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

CONSTRUCT_PATTERNS = {
    "goal_setting": [
        re.compile(
            r"\b(my goal is|goal is to|i want to|i need to|objective is to|aim to)\b[^.?!]{0,120}",
            flags=re.IGNORECASE,
        )
    ],
    "goal_navigation": [
        re.compile(
            r"\b(next|then|after that|first[, ]+|second[, ]+|priority|prioritize)\b[^.?!]{0,120}",
            flags=re.IGNORECASE,
        )
    ],
    "problem_decomposition": [
        re.compile(
            r"\b(break (it|this) down|split (it|this) into|step by step|components?|sub[- ]?tasks?)\b[^.?!]{0,120}",
            flags=re.IGNORECASE,
        )
    ],
    "strategic_planning": [
        re.compile(
            r"\b(strategy|approach|roadmap|plan to|planning to|best way to)\b[^.?!]{0,120}",
            flags=re.IGNORECASE,
        )
    ],
    "progress_monitoring": [
        re.compile(
            r"\b(progress|milestone|track(ing)?|status|measure|checkpoint)\b[^.?!]{0,120}",
            flags=re.IGNORECASE,
        )
    ],
    "obstacle_management": [
        re.compile(
            r"\b(stuck|blocked|error|issue|bug|constraint|difficulty|challenge)\b[^.?!]{0,120}",
            flags=re.IGNORECASE,
        )
    ],
    "help_seeking_resourcefulness": [
        re.compile(
            r"\b(help me|can you|show me|recommend|resources?|reference|example code)\b[^.?!]{0,120}",
            flags=re.IGNORECASE,
        )
    ],
    "self_efficacy_or_confidence": [
        re.compile(
            r"\b(i can|i am able to|i feel confident|i know how to|i will be able to)\b[^.?!]{0,120}",
            flags=re.IGNORECASE,
        )
    ],
    "resilience_or_persistence": [
        re.compile(
            r"\b(try again|keep going|still working on|despite|even though|did not work so|iterate)\b[^.?!]{0,120}",
            flags=re.IGNORECASE,
        )
    ],
}

EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", flags=re.IGNORECASE)
URL_PATTERN = re.compile(r"\b(?:https?://|www\.)\S+\b", flags=re.IGNORECASE)
PHONE_PATTERN = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?!\w)"
)
LONG_ID_PATTERN = re.compile(
    r"\b[a-f0-9]{16,}\b|\b[A-Za-z0-9_-]{20,}\b",
    flags=re.IGNORECASE,
)


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
        "--min-chars",
        type=int,
        default=DEFAULT_MIN_SNIPPET_CHARS,
        help="Minimum snippet size for sampling tasks.",
    )
    parser.add_argument(
        "--model-id",
        default="gpt-4o-mini",
        help="Model identifier (used by extraction scripts).",
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


def pick_snippet(text: str, min_chars: int = DEFAULT_MIN_SNIPPET_CHARS, max_chars: int = MAX_SNIPPET_CHARS) -> str:
    cleaned = normalize_whitespace(text)
    if len(cleaned) <= max_chars:
        return cleaned

    candidate = cleaned[:max_chars]
    boundary_candidates = [candidate.rfind(". "), candidate.rfind("? "), candidate.rfind("! "), candidate.rfind("; ")]
    best_boundary = max(boundary_candidates)
    if best_boundary >= min_chars:
        return candidate[: best_boundary + 1].strip()
    return candidate.strip()


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


def heuristic_construct_extractions(text: str, max_spans_per_construct: int = 1) -> list[dict[str, Any]]:
    extractions: list[dict[str, Any]] = []
    for construct in CONSTRUCTS:
        patterns = CONSTRUCT_PATTERNS.get(construct, [])
        added = 0
        for pattern in patterns:
            for match in pattern.finditer(text):
                span = match.group(0).strip()
                if not span:
                    continue
                extractions.append(
                    {
                        "extraction_class": construct,
                        "extraction_text": span,
                        "char_start": match.start(),
                        "char_end": match.end(),
                        "attributes": {"seed_source": "heuristic_example_bootstrap"},
                    }
                )
                added += 1
                if added >= max_spans_per_construct:
                    break
            if added >= max_spans_per_construct:
                break
    return extractions


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

