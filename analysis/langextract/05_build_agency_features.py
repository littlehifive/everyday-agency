from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import polars as pl

from agency_common import CONSTRUCTS, add_common_args, ensure_parent_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build conversation-level binary and count agency features."
    )
    add_common_args(
        parser,
        default_input="data/derived/langextract/langextract_extractions_long.parquet",
        default_output="data/derived/langextract/conversation_agency_features.parquet",
    )
    parser.add_argument(
        "--metadata",
        default="data/derived/langextract/conversations_candidates.parquet",
        help="Candidate metadata parquet path.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/derived/langextract/conversation_agency_features.csv",
        help="Output CSV path for conversation-level features.",
    )
    parser.add_argument(
        "--output-prevalence",
        default="data/derived/langextract/construct_prevalence.parquet",
        help="Output parquet path for construct prevalence summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata)
    extraction_path = Path(args.input)
    output_parquet_path = Path(args.output)
    output_csv_path = Path(args.output_csv)
    prevalence_output_path = Path(args.output_prevalence)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    if not extraction_path.exists():
        raise FileNotFoundError(f"Missing extraction file: {extraction_path}")

    metadata_df = pl.read_parquet(metadata_path)
    if args.limit is not None:
        metadata_df = metadata_df.head(args.limit)

    if metadata_df.select(pl.col("conversation_id").n_unique()).item() != metadata_df.height:
        raise ValueError("Metadata must contain one row per conversation_id.")

    metadata_pd = metadata_df.to_pandas()
    extraction_pd = pd.read_parquet(extraction_path)

    extraction_pd = extraction_pd[extraction_pd["extraction_class"].isin(CONSTRUCTS)]
    if args.limit is not None:
        valid_ids = set(metadata_pd["conversation_id"].tolist())
        extraction_pd = extraction_pd[extraction_pd["conversation_id"].isin(valid_ids)]

    if extraction_pd.empty:
        counts = pd.DataFrame(index=metadata_pd["conversation_id"])
    else:
        counts = (
            extraction_pd.groupby(["conversation_id", "extraction_class"])
            .size()
            .unstack(fill_value=0)
        )
    counts = counts.reindex(index=metadata_pd["conversation_id"], fill_value=0)

    features = metadata_pd.copy()
    for construct in CONSTRUCTS:
        count_col = f"{construct}_count"
        flag_col = f"{construct}_flag"
        series = counts[construct] if construct in counts.columns else pd.Series(0, index=counts.index)
        series = series.reindex(features["conversation_id"]).fillna(0).astype(int).reset_index(drop=True)
        features[count_col] = series
        features[flag_col] = (features[count_col] > 0).astype(int)

    flag_cols = [f"{construct}_flag" for construct in CONSTRUCTS]
    count_cols = [f"{construct}_count" for construct in CONSTRUCTS]

    features["agency_total_flags"] = features[flag_cols].sum(axis=1)
    features["agency_total_count"] = features[count_cols].sum(axis=1)
    features = features.sort_values(["agency_total_count", "agency_total_flags"], ascending=False)

    if not features["conversation_id"].is_unique:
        raise ValueError("Final feature table contains duplicate conversation_id values.")
    for col in flag_cols:
        if not set(features[col].unique()).issubset({0, 1}):
            raise ValueError(f"Non-binary flag values found in {col}")
    for construct in CONSTRUCTS:
        count_col = f"{construct}_count"
        flag_col = f"{construct}_flag"
        mismatch = features[(features[count_col] > 0) & (features[flag_col] != 1)]
        if not mismatch.empty:
            raise ValueError(f"Found count/flag mismatch for {construct}")

    prevalence_rows = []
    n_conversations = len(features)
    for construct in CONSTRUCTS:
        count_col = f"{construct}_count"
        flag_col = f"{construct}_flag"
        n_flagged = int(features[flag_col].sum())
        prevalence_rows.append(
            {
                "construct": construct,
                "n_conversations": int(n_conversations),
                "n_flagged_conversations": n_flagged,
                "prevalence_rate": float(n_flagged / max(n_conversations, 1)),
                "total_instances": int(features[count_col].sum()),
            }
        )
    prevalence_df = pl.DataFrame(prevalence_rows).sort("prevalence_rate", descending=True)

    ensure_parent_dir(output_parquet_path)
    pl.from_pandas(features).write_parquet(output_parquet_path)

    ensure_parent_dir(output_csv_path)
    features.to_csv(output_csv_path, index=False)

    ensure_parent_dir(prevalence_output_path)
    prevalence_df.write_parquet(prevalence_output_path)

    print(f"Wrote conversation-level feature parquet: {output_parquet_path}")
    print(f"Wrote conversation-level feature csv: {output_csv_path}")
    print(f"Wrote prevalence summary parquet: {prevalence_output_path}")


if __name__ == "__main__":
    main()

