from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import polars as pl

from gabriel_common import (
    AGENCY_ATTRIBUTES,
    add_common_args,
    ensure_parent_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build conversation-level Gabriel agency features and composite indices."
    )
    add_common_args(
        parser,
        default_input="data/derived/gabriel/conversation_agency_ratings_wide.parquet",
        default_output="data/derived/gabriel/conversation_agency_features.parquet",
    )
    parser.add_argument(
        "--metadata",
        default="data/derived/gabriel/conversations_candidates.parquet",
        help="Conversation metadata parquet path.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/derived/gabriel/conversation_agency_features.csv",
        help="Output CSV path for conversation-level features.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ratings_path = Path(args.input)
    metadata_path = Path(args.metadata)
    output_parquet_path = Path(args.output)
    output_csv_path = Path(args.output_csv)

    if not ratings_path.exists():
        raise FileNotFoundError(f"Missing ratings file: {ratings_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    ratings_df = pl.read_parquet(ratings_path)
    metadata_df = pl.read_parquet(metadata_path)

    if args.limit is not None:
        valid_ids = set(metadata_df.head(args.limit)["conversation_id"].to_list())
        ratings_df = ratings_df.filter(pl.col("conversation_id").is_in(valid_ids))
        metadata_df = metadata_df.head(args.limit)

    missing_rating_columns = [c for c in AGENCY_ATTRIBUTES if c not in ratings_df.columns]
    if missing_rating_columns:
        raise ValueError(f"Missing agency rating columns: {missing_rating_columns}")

    feature_pd = ratings_df.to_pandas()
    metadata_pd = metadata_df.select(
        [
            "conversation_id",
            "conversation_title",
            "n_user_messages",
            "n_user_chars",
            "first_user_time",
            "last_user_time",
            "title_score",
            "text_cue_score",
            "is_candidate",
            "candidate_reasons",
        ]
    ).to_pandas()

    feature_pd = feature_pd.merge(
        metadata_pd, on="conversation_id", how="left", suffixes=("", "_meta")
    )
    feature_pd["agency_index_raw"] = feature_pd[AGENCY_ATTRIBUTES].mean(axis=1)
    feature_pd["agency_spread"] = (
        feature_pd[AGENCY_ATTRIBUTES].max(axis=1) - feature_pd[AGENCY_ATTRIBUTES].min(axis=1)
    )
    feature_pd["proxy_minus_personal"] = (
        feature_pd["proxy_agency"] - feature_pd["personal_agency"]
    )

    mean_raw = float(feature_pd["agency_index_raw"].mean())
    std_raw = float(feature_pd["agency_index_raw"].std(ddof=0))
    if std_raw == 0:
        feature_pd["agency_index_z"] = 0.0
    else:
        feature_pd["agency_index_z"] = (feature_pd["agency_index_raw"] - mean_raw) / std_raw

    feature_pd["agency_high_flag"] = (feature_pd["agency_index_z"] >= 1.0).astype(int)
    feature_pd["agency_low_flag"] = (feature_pd["agency_index_z"] <= -1.0).astype(int)
    feature_pd = feature_pd.sort_values(
        ["agency_index_raw", "personal_agency"], ascending=[False, False]
    )

    if not feature_pd["conversation_id"].is_unique:
        raise ValueError("Final feature table contains duplicate conversation_id values.")

    for col in AGENCY_ATTRIBUTES:
        if feature_pd[col].min() < 0 or feature_pd[col].max() > 100:
            raise ValueError(f"Attribute out of range [0,100]: {col}")

    ensure_parent_dir(output_parquet_path)
    pl.from_pandas(feature_pd).write_parquet(output_parquet_path)
    ensure_parent_dir(output_csv_path)
    feature_pd.to_csv(output_csv_path, index=False)

    print(f"Wrote conversation-level features parquet: {output_parquet_path}")
    print(f"Wrote conversation-level features csv: {output_csv_path}")


if __name__ == "__main__":
    main()
