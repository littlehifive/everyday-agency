from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from agency_common import (
    TEXT_JOIN_DELIMITER,
    add_common_args,
    clean_title,
    ensure_parent_dir,
    normalize_whitespace,
)

REQUIRED_COLUMNS = {
    "conversation_id",
    "conversation_title",
    "author_role",
    "create_time",
    "text",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a conversation-level corpus using user-authored messages."
    )
    add_common_args(
        parser,
        default_input="data/chat_messages.parquet",
        default_output="data/derived/agency/conversations_user.parquet",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    df = pl.read_parquet(input_path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    user_df = (
        df.filter(pl.col("author_role") == "user")
        .with_columns(
            [
                pl.col("text")
                .map_elements(normalize_whitespace, return_dtype=pl.String)
                .alias("text_clean"),
                pl.col("conversation_title")
                .map_elements(clean_title, return_dtype=pl.String)
                .alias("title_clean"),
            ]
        )
        .filter(pl.col("text_clean").str.len_chars() > 0)
        .sort(["conversation_id", "create_time"])
    )

    aggregated = (
        user_df.group_by("conversation_id")
        .agg(
            [
                pl.col("title_clean").drop_nulls().first().alias("conversation_title"),
                pl.col("text_clean").str.join(TEXT_JOIN_DELIMITER).alias("user_text_concat"),
                pl.len().alias("n_user_messages"),
                pl.col("text_clean").str.len_chars().sum().alias("n_user_chars"),
                pl.col("create_time").min().alias("first_user_time"),
                pl.col("create_time").max().alias("last_user_time"),
            ]
        )
        .sort("first_user_time")
    )

    if args.limit is not None:
        aggregated = aggregated.head(args.limit)

    ensure_parent_dir(output_path)
    aggregated.write_parquet(output_path)

    print(f"Wrote {aggregated.height} rows to {output_path}")
    print(f"Columns: {aggregated.columns}")


if __name__ == "__main__":
    main()
