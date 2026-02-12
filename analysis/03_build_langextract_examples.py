from __future__ import annotations

import argparse
import random
from pathlib import Path

import polars as pl

from agency_common import (
    CONSTRUCTS,
    CONSTRUCT_DEFINITIONS,
    add_common_args,
    ensure_parent_dir,
    heuristic_construct_extractions,
    pick_snippet,
    redact_text,
    write_json,
)

REQUIRED_COLUMNS = {
    "conversation_id",
    "conversation_title",
    "user_text_concat",
    "is_candidate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create redacted snippets and few-shot examples for LangExtract agency coding."
    )
    add_common_args(
        parser,
        default_input="data/derived/agency/conversations_candidates.parquet",
        default_output="data/derived/agency/example_snippets_redacted.csv",
    )
    parser.add_argument(
        "--prompt-output",
        default="analysis/config/agency_prompt.md",
        help="Path to write prompt instructions.",
    )
    parser.add_argument(
        "--examples-output",
        default="analysis/config/agency_examples.json",
        help="Path to write LangExtract few-shot examples.",
    )
    return parser.parse_args()


def build_prompt_text() -> str:
    definitions = "\n".join(
        [f"- `{name}`: {definition}" for name, definition in CONSTRUCT_DEFINITIONS.items()]
    )
    return (
        "# Agency Signal Extraction Task\n\n"
        "Extract short spans that indicate everyday agency in user-authored text.\n\n"
        "## Valid extraction classes\n"
        f"{definitions}\n\n"
        "## Rules\n"
        "- Extract only text that appears verbatim in the input.\n"
        "- Keep spans minimal while preserving meaning.\n"
        "- Use only the listed classes.\n"
        "- Include multiple classes if the text clearly supports them.\n"
        "- If no agency signal is present, return no extractions.\n"
    )


def sample_records(records: list[dict], n: int, seed: int) -> list[dict]:
    if n <= 0 or not records:
        return []
    rng = random.Random(seed)
    if len(records) <= n:
        return list(records)
    indices = rng.sample(range(len(records)), n)
    return [records[i] for i in indices]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    csv_output_path = Path(args.output)
    prompt_output_path = Path(args.prompt_output)
    examples_output_path = Path(args.examples_output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    df = pl.read_parquet(input_path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if args.limit is not None:
        df = df.head(args.limit)

    snippet_records: list[dict] = []
    for row in df.iter_rows(named=True):
        snippet = pick_snippet(
            row["user_text_concat"] or "",
            min_chars=args.min_chars,
            max_chars=700,
        )
        redacted = redact_text(snippet)
        if len(redacted) < max(80, args.min_chars // 3):
            continue

        exts = heuristic_construct_extractions(redacted, max_spans_per_construct=1)
        construct_set = sorted({entry["extraction_class"] for entry in exts})
        snippet_records.append(
            {
                "conversation_id": row["conversation_id"],
                "conversation_title": row["conversation_title"],
                "is_candidate": bool(row["is_candidate"]),
                "snippet_redacted": redacted,
                "snippet_char_len": len(redacted),
                "heuristic_extractions": exts,
                "heuristic_constructs": construct_set,
                "n_heuristic_extractions": len(exts),
            }
        )

    pos_records = [r for r in snippet_records if r["is_candidate"]]
    neg_records = [r for r in snippet_records if not r["is_candidate"]]

    selected_pos = sample_records(pos_records, 24, seed=args.seed)
    selected_neg = sample_records(neg_records, 6, seed=args.seed + 1)

    selected_ids = {r["conversation_id"] for r in selected_pos + selected_neg}
    if len(selected_pos) + len(selected_neg) < 30:
        remainder = [r for r in snippet_records if r["conversation_id"] not in selected_ids]
        filler = sample_records(remainder, 30 - len(selected_pos) - len(selected_neg), seed=args.seed + 2)
    else:
        filler = []

    selected = selected_pos + selected_neg + filler

    csv_rows = []
    for row in selected:
        csv_rows.append(
            {
                "conversation_id": row["conversation_id"],
                "conversation_title": row["conversation_title"],
                "is_candidate": row["is_candidate"],
                "snippet_char_len": row["snippet_char_len"],
                "n_heuristic_extractions": row["n_heuristic_extractions"],
                "heuristic_constructs": ",".join(row["heuristic_constructs"]),
                "snippet_redacted": row["snippet_redacted"],
            }
        )
    csv_df = pl.DataFrame(csv_rows).sort(
        ["is_candidate", "n_heuristic_extractions", "snippet_char_len"],
        descending=[True, True, True],
    )

    ensure_parent_dir(csv_output_path)
    csv_df.write_csv(csv_output_path)

    positive_candidates = [r for r in selected if r["n_heuristic_extractions"] > 0]
    negative_candidates = [r for r in selected if r["n_heuristic_extractions"] == 0]

    selected_example_records: list[dict] = []
    used_conversation_ids: set[str] = set()

    for construct in CONSTRUCTS:
        eligible = [
            r
            for r in positive_candidates
            if construct in r["heuristic_constructs"] and r["conversation_id"] not in used_conversation_ids
        ]
        if not eligible:
            continue
        best = sorted(eligible, key=lambda item: item["n_heuristic_extractions"], reverse=True)[0]
        selected_example_records.append(best)
        used_conversation_ids.add(best["conversation_id"])

    if len(selected_example_records) < 12:
        n_to_add = 12 - len(selected_example_records)
        remainder_positive = [
            r for r in positive_candidates if r["conversation_id"] not in used_conversation_ids
        ]
        remainder_positive = sorted(
            remainder_positive,
            key=lambda item: item["n_heuristic_extractions"],
            reverse=True,
        )
        additions = remainder_positive[:n_to_add]
        selected_example_records.extend(additions)
        used_conversation_ids.update(r["conversation_id"] for r in additions)

    negative_examples = sorted(
        negative_candidates,
        key=lambda item: item["snippet_char_len"],
        reverse=True,
    )[:2]
    selected_example_records.extend(negative_examples)

    examples_payload = {
        "schema_version": 1,
        "constructs": CONSTRUCTS,
        "examples": [
            {
                "text": row["snippet_redacted"],
                "extractions": [
                    {
                        "extraction_class": ext["extraction_class"],
                        "extraction_text": ext["extraction_text"],
                        "attributes": ext.get("attributes", {}),
                    }
                    for ext in row["heuristic_extractions"]
                ],
                "metadata": {
                    "conversation_id": row["conversation_id"],
                    "conversation_title": row["conversation_title"],
                    "is_candidate": row["is_candidate"],
                },
            }
            for row in selected_example_records
        ],
    }

    ensure_parent_dir(prompt_output_path)
    prompt_output_path.write_text(build_prompt_text(), encoding="utf-8")
    write_json(examples_output_path, examples_payload)

    print(f"Wrote {csv_df.height} redacted snippets to {csv_output_path}")
    print(f"Wrote prompt to {prompt_output_path}")
    print(
        "Wrote "
        f"{len(examples_payload['examples'])} few-shot examples to {examples_output_path}"
    )


if __name__ == "__main__":
    main()
