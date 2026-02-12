from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import langextract as lx
import polars as pl

from agency_common import CONSTRUCTS, add_common_args, ensure_parent_dir, load_env_file

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
        description="Run LangExtract agency coding on candidate conversations."
    )
    add_common_args(
        parser,
        default_input="data/derived/agency/conversations_candidates.parquet",
        default_output="data/derived/agency/langextract_extractions_long.parquet",
    )
    parser.add_argument(
        "--prompt-file",
        default="analysis/config/agency_prompt.md",
        help="Prompt description markdown path.",
    )
    parser.add_argument(
        "--examples-file",
        default="analysis/config/agency_examples.json",
        help="Few-shot example JSON path.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="data/derived/agency/langextract_annotations.jsonl",
        help="Path for raw LangExtract annotations JSONL.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM sampling temperature.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="LangExtract worker threads.",
    )
    parser.add_argument(
        "--batch-length",
        type=int,
        default=8,
        help="LangExtract batch length.",
    )
    parser.add_argument(
        "--extraction-passes",
        type=int,
        default=1,
        help="Number of extraction passes.",
    )
    return parser.parse_args()


def load_examples(examples_path: Path) -> list[lx.data.ExampleData]:
    payload = json.loads(examples_path.read_text(encoding="utf-8"))
    if not payload:
        return []
    root = payload
    examples: list[lx.data.ExampleData] = []
    for item in root.get("examples", []):
        extractions = [
            lx.data.Extraction(
                extraction_class=ext["extraction_class"],
                extraction_text=ext["extraction_text"],
                attributes=ext.get("attributes"),
            )
            for ext in item.get("extractions", [])
        ]
        examples.append(lx.data.ExampleData(text=item["text"], extractions=extractions))
    return examples


def empty_extraction_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "conversation_id": pl.Series([], dtype=pl.String),
            "conversation_title": pl.Series([], dtype=pl.String),
            "extraction_class": pl.Series([], dtype=pl.String),
            "extraction_text": pl.Series([], dtype=pl.String),
            "extraction_index": pl.Series([], dtype=pl.Int64),
            "group_index": pl.Series([], dtype=pl.Int64),
            "char_start": pl.Series([], dtype=pl.Int64),
            "char_end": pl.Series([], dtype=pl.Int64),
            "alignment_status": pl.Series([], dtype=pl.String),
            "attributes_json": pl.Series([], dtype=pl.String),
        }
    )


def main() -> None:
    args = parse_args()
    load_env_file(".env")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it to env or .env before running extraction."
        )

    input_path = Path(args.input)
    prompt_path = Path(args.prompt_file)
    examples_path = Path(args.examples_file)
    output_long_path = Path(args.output)
    output_jsonl_path = Path(args.output_jsonl)

    for required_file in [input_path, prompt_path, examples_path]:
        if not required_file.exists():
            raise FileNotFoundError(f"Required file not found: {required_file}")

    df = pl.read_parquet(input_path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.filter(pl.col("is_candidate"))
    if args.limit is not None:
        df = df.head(args.limit)

    if df.is_empty():
        ensure_parent_dir(output_long_path)
        empty_extraction_df().write_parquet(output_long_path)
        ensure_parent_dir(output_jsonl_path)
        output_jsonl_path.write_text("", encoding="utf-8")
        print("No candidate conversations found. Wrote empty outputs.")
        return

    prompt_text = prompt_path.read_text(encoding="utf-8")
    examples = load_examples(examples_path)

    metadata_by_conversation_id = {
        row["conversation_id"]: {
            "conversation_title": row["conversation_title"],
        }
        for row in df.select(["conversation_id", "conversation_title"]).iter_rows(named=True)
    }

    documents = [
        lx.data.Document(
            text=row["user_text_concat"],
            document_id=row["conversation_id"],
            additional_context=f"conversation_title: {row['conversation_title']}",
        )
        for row in df.select(["conversation_id", "conversation_title", "user_text_concat"]).iter_rows(named=True)
    ]

    result = lx.extract(
        documents,
        prompt_description=prompt_text,
        examples=examples,
        model_id=args.model_id,
        api_key=api_key,
        temperature=args.temperature,
        max_workers=args.max_workers,
        batch_length=args.batch_length,
        extraction_passes=args.extraction_passes,
        fetch_urls=False,
        prompt_validation_level=lx.prompt_validation.PromptValidationLevel.OFF,
        show_progress=True,
    )

    if isinstance(result, list):
        annotated_documents = result
    else:
        annotated_documents = [result]

    ensure_parent_dir(output_jsonl_path)
    lx.io.save_annotated_documents(
        annotated_documents,
        output_dir=output_jsonl_path.parent,
        output_name=output_jsonl_path.name,
        show_progress=True,
    )

    rows: list[dict] = []
    for doc in annotated_documents:
        conversation_id = doc.document_id
        conversation_title = metadata_by_conversation_id.get(conversation_id, {}).get(
            "conversation_title",
            "",
        )
        extractions = doc.extractions or []
        for extraction in extractions:
            alignment = extraction.alignment_status
            if hasattr(alignment, "value"):
                alignment = alignment.value
            char_start = (
                extraction.char_interval.start_pos
                if extraction.char_interval is not None
                else None
            )
            char_end = (
                extraction.char_interval.end_pos
                if extraction.char_interval is not None
                else None
            )
            rows.append(
                {
                    "conversation_id": conversation_id,
                    "conversation_title": conversation_title,
                    "extraction_class": extraction.extraction_class,
                    "extraction_text": extraction.extraction_text,
                    "extraction_index": extraction.extraction_index,
                    "group_index": extraction.group_index,
                    "char_start": char_start,
                    "char_end": char_end,
                    "alignment_status": alignment,
                    "attributes_json": json.dumps(extraction.attributes or {}),
                }
            )

    extraction_df = pl.DataFrame(rows) if rows else empty_extraction_df()
    extraction_df = extraction_df.filter(pl.col("extraction_class").is_in(CONSTRUCTS))

    ensure_parent_dir(output_long_path)
    extraction_df.write_parquet(output_long_path)

    print(f"Coded documents: {len(annotated_documents)}")
    print(f"Extracted rows: {extraction_df.height}")
    print(f"Wrote JSONL annotations: {output_jsonl_path}")
    print(f"Wrote long extraction parquet: {output_long_path}")


if __name__ == "__main__":
    main()
