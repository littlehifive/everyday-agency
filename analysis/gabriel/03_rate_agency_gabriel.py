from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from gabriel_common import (
    AGENCY_ATTRIBUTES,
    add_common_args,
    build_deterministic_chunks,
    ensure_parent_dir,
    load_env_file,
    read_json,
)

REQUIRED_COLUMNS = {
    "conversation_id",
    "conversation_title",
    "user_text_concat",
    "n_user_messages",
    "n_user_chars",
    "title_score",
    "text_cue_score",
    "is_candidate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Gabriel agency ratings on candidate conversations."
    )
    add_common_args(
        parser,
        default_input="data/derived/gabriel/conversations_candidates.parquet",
        default_output="data/derived/gabriel/conversation_agency_ratings_wide.parquet",
    )
    parser.add_argument(
        "--output-long",
        default="data/derived/gabriel/conversation_agency_ratings_long.parquet",
        help="Long-format rating output path.",
    )
    parser.add_argument(
        "--attributes-file",
        default="analysis/gabriel/config/agency_attributes_v1.json",
        help="JSON file containing Gabriel agency attributes.",
    )
    parser.add_argument(
        "--instructions-file",
        default="analysis/gabriel/config/agency_additional_instructions_v1.md",
        help="Markdown file with additional rating instructions.",
    )
    parser.add_argument(
        "--template-path",
        default=None,
        help="Optional custom Jinja template path for gabriel.rate.",
    )
    parser.add_argument(
        "--candidate-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only score rows where is_candidate is true.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=2,
        help="Number of gabriel.rate runs to aggregate.",
    )
    parser.add_argument(
        "--n-parallels",
        type=int,
        default=150,
        help="Max parallel requests for Gabriel.",
    )
    parser.add_argument(
        "--n-attributes-per-run",
        type=int,
        default=8,
        help="Attributes per prompt batch in gabriel.rate.",
    )
    parser.add_argument(
        "--max-chars-per-chunk",
        type=int,
        default=6000,
        help="Max characters per text chunk.",
    )
    parser.add_argument(
        "--max-chunks-per-conversation",
        type=int,
        default=5,
        help="Cap scored chunks per conversation.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="low",
        help="Gabriel reasoning effort passed to model.",
    )
    parser.add_argument(
        "--reset-files",
        action="store_true",
        help="Force regeneration of Gabriel checkpoint files.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="data/derived/gabriel/rate_checkpoints",
        help="Directory for gabriel.rate checkpoints and raw responses.",
    )
    parser.add_argument("--rubric-version", default="v1", help="Rubric version label.")
    parser.add_argument(
        "--instruction-version", default="v1", help="Instruction version label."
    )
    parser.add_argument(
        "--template-version", default="default", help="Template version label."
    )
    return parser.parse_args()


def load_attributes(attributes_path: Path) -> dict[str, str]:
    payload = read_json(attributes_path)
    if "attributes" in payload and isinstance(payload["attributes"], dict):
        attrs = payload["attributes"]
    elif isinstance(payload, dict):
        attrs = payload
    else:
        raise ValueError("Attributes file must contain a JSON object.")
    if not attrs:
        raise ValueError("Attributes file is empty.")
    missing_expected = [name for name in AGENCY_ATTRIBUTES if name not in attrs]
    if missing_expected:
        raise ValueError(f"Missing expected agency attributes: {missing_expected}")
    return {str(k): str(v) for k, v in attrs.items()}


def build_chunk_dataframe(
    df: pl.DataFrame,
    *,
    max_chars_per_chunk: int,
    max_chunks_per_conversation: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in df.iter_rows(named=True):
        chunks, n_chunks_total, coverage = build_deterministic_chunks(
            row["user_text_concat"] or "",
            max_chars_per_chunk=max_chars_per_chunk,
            max_chunks=max_chunks_per_conversation,
        )
        if not chunks:
            continue
        for chunk_idx, chunk_text in enumerate(chunks):
            rows.append(
                {
                    "conversation_id": row["conversation_id"],
                    "conversation_title": row["conversation_title"],
                    "chunk_index": chunk_idx,
                    "chunk_id": f"{row['conversation_id']}__chunk_{chunk_idx:03d}",
                    "chunk_text": chunk_text,
                    "chunk_char_len": len(chunk_text),
                    "n_chunks_total": n_chunks_total,
                    "n_chunks_scored": len(chunks),
                    "char_coverage_ratio": float(coverage),
                    "n_user_messages": int(row["n_user_messages"]),
                    "n_user_chars": int(row["n_user_chars"]),
                    "is_candidate": bool(row["is_candidate"]),
                }
            )
    return pd.DataFrame(rows)


def to_long_df(
    rated_df: pd.DataFrame,
    *,
    attributes: list[str],
    model_id: str,
    rubric_version: str,
    instruction_version: str,
    template_version: str,
) -> pl.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rated_df.to_dict(orient="records"):
        for attribute in attributes:
            score = row.get(attribute)
            if pd.isna(score):
                continue
            records.append(
                {
                    "conversation_id": row["conversation_id"],
                    "chunk_id": row["chunk_id"],
                    "chunk_index": int(row["chunk_index"]),
                    "attribute": attribute,
                    "score": float(score),
                    "run_index": 0,
                    "model_id": model_id,
                    "rubric_version": rubric_version,
                    "instruction_version": instruction_version,
                    "template_version": template_version,
                    "is_candidate": bool(row["is_candidate"]),
                    "n_chunks_total": int(row["n_chunks_total"]),
                    "n_chunks_scored": int(row["n_chunks_scored"]),
                    "char_coverage_ratio": float(row["char_coverage_ratio"]),
                }
            )
    return pl.DataFrame(records)


def build_wide_df(
    rated_df: pd.DataFrame,
    *,
    attributes: list[str],
    model_id: str,
    rubric_version: str,
    instruction_version: str,
    template_version: str,
) -> pl.DataFrame:
    attribute_agg: dict[str, Any] = {attribute: "mean" for attribute in attributes}
    agg = (
        rated_df.groupby("conversation_id", as_index=False)
        .agg(
            {
                **attribute_agg,
                "conversation_title": "first",
                "n_user_messages": "first",
                "n_user_chars": "first",
                "is_candidate": "first",
                "n_chunks_total": "max",
                "n_chunks_scored": "max",
                "char_coverage_ratio": "max",
            }
        )
        .sort_values("conversation_id")
    )
    agg["model_id"] = model_id
    agg["rubric_version"] = rubric_version
    agg["instruction_version"] = instruction_version
    agg["template_version"] = template_version
    return pl.from_pandas(agg)


async def run_rate(args: argparse.Namespace) -> None:
    try:
        import gabriel
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "gabriel package not found in this interpreter. Activate the virtual environment "
            "where openai-gabriel is installed before running this script."
        ) from exc

    load_env_file(".env")
    input_path = Path(args.input)
    output_wide_path = Path(args.output)
    output_long_path = Path(args.output_long)
    attributes_path = Path(args.attributes_file)
    instructions_path = Path(args.instructions_file)
    checkpoint_dir = Path(args.checkpoint_dir)

    for required_path in [input_path, attributes_path, instructions_path]:
        if not required_path.exists():
            raise FileNotFoundError(f"Required file not found: {required_path}")

    df = pl.read_parquet(input_path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if args.candidate_only:
        df = df.filter(pl.col("is_candidate"))
    if args.limit is not None:
        df = df.head(args.limit)
    if df.is_empty():
        raise ValueError("No rows available for scoring after filters.")

    attributes_map = load_attributes(attributes_path)
    additional_instructions = instructions_path.read_text(encoding="utf-8")
    chunk_df = build_chunk_dataframe(
        df,
        max_chars_per_chunk=args.max_chars_per_chunk,
        max_chunks_per_conversation=args.max_chunks_per_conversation,
    )
    if chunk_df.empty:
        raise ValueError("No chunks were generated from input conversations.")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rated = await gabriel.rate(
        df=chunk_df,
        column_name="chunk_text",
        attributes=attributes_map,
        save_dir=str(checkpoint_dir),
        model=args.model_id,
        n_runs=args.n_runs,
        n_parallels=args.n_parallels,
        n_attributes_per_run=args.n_attributes_per_run,
        additional_instructions=additional_instructions,
        modality="text",
        reasoning_effort=args.reasoning_effort,
        reset_files=args.reset_files,
        template_path=args.template_path,
        file_name="conversation_chunk_ratings.csv",
    )

    attributes = list(attributes_map.keys())
    for attr in attributes:
        rated[attr] = pd.to_numeric(rated[attr], errors="coerce")

    long_df = to_long_df(
        rated,
        attributes=attributes,
        model_id=args.model_id,
        rubric_version=args.rubric_version,
        instruction_version=args.instruction_version,
        template_version=args.template_version,
    )
    wide_df = build_wide_df(
        rated,
        attributes=attributes,
        model_id=args.model_id,
        rubric_version=args.rubric_version,
        instruction_version=args.instruction_version,
        template_version=args.template_version,
    )

    ensure_parent_dir(output_long_path)
    long_df.write_parquet(output_long_path)
    ensure_parent_dir(output_wide_path)
    wide_df.write_parquet(output_wide_path)

    metadata_path = output_wide_path.parent / "conversation_agency_rating_metadata.json"
    metadata_payload = {
        "input_path": str(input_path),
        "output_wide_path": str(output_wide_path),
        "output_long_path": str(output_long_path),
        "checkpoint_dir": str(checkpoint_dir),
        "candidate_only": bool(args.candidate_only),
        "n_conversations_scored": int(wide_df.height),
        "n_chunks_scored": int(rated.shape[0]),
        "attributes": attributes,
        "model_id": args.model_id,
        "n_runs": int(args.n_runs),
        "n_parallels": int(args.n_parallels),
        "n_attributes_per_run": int(args.n_attributes_per_run),
        "max_chars_per_chunk": int(args.max_chars_per_chunk),
        "max_chunks_per_conversation": int(args.max_chunks_per_conversation),
        "rubric_version": args.rubric_version,
        "instruction_version": args.instruction_version,
        "template_version": args.template_version,
        "template_path": args.template_path,
    }
    ensure_parent_dir(metadata_path)
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")

    print(f"Scored conversations: {wide_df.height}")
    print(f"Scored chunks: {rated.shape[0]}")
    print(f"Wrote long ratings: {output_long_path}")
    print(f"Wrote wide ratings: {output_wide_path}")
    print(f"Wrote metadata: {metadata_path}")


def main() -> None:
    args = parse_args()
    asyncio.run(run_rate(args))


if __name__ == "__main__":
    main()
