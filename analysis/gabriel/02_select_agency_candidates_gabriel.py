from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from gabriel_common import add_common_args, ensure_parent_dir, score_text_cues, score_title

REQUIRED_COLUMNS = {
    "conversation_id",
    "conversation_title",
    "user_text_concat",
    "n_user_messages",
    "n_user_chars",
    "first_user_time",
    "last_user_time",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply two-stage filtering for conversations likely to show agency signals."
    )
    add_common_args(
        parser,
        default_input="data/derived/gabriel/conversations_user.parquet",
        default_output="data/derived/gabriel/conversations_candidates.parquet",
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

    if args.limit is not None:
        df = df.head(args.limit)

    records: list[dict] = []
    for row in df.iter_rows(named=True):
        title_score, matched_title_terms = score_title(row["conversation_title"] or "")
        text_score, matched_text_cues = score_text_cues(row["user_text_concat"] or "")

        is_candidate = (
            (title_score >= 2)
            or (text_score >= 3)
            or ((text_score >= 2) and (int(row["n_user_chars"]) >= 400))
        )

        reasons: list[str] = []
        if title_score >= 2:
            reasons.append("title_score>=2")
        if text_score >= 3:
            reasons.append("text_cue_score>=3")
        if (text_score >= 2) and (int(row["n_user_chars"]) >= 400):
            reasons.append("text_cue_score>=2_and_n_user_chars>=400")

        row_out = dict(row)
        row_out["title_score"] = int(title_score)
        row_out["text_cue_score"] = int(text_score)
        row_out["is_candidate"] = bool(is_candidate)
        row_out["candidate_reasons"] = ";".join(reasons) if reasons else "none"
        row_out["matched_title_terms"] = matched_title_terms
        row_out["matched_text_cues"] = matched_text_cues
        records.append(row_out)

    out_df = pl.DataFrame(records).sort(
        ["is_candidate", "text_cue_score", "title_score", "n_user_chars"],
        descending=[True, True, True, True],
    )

    ensure_parent_dir(output_path)
    out_df.write_parquet(output_path)

    candidate_count = out_df.filter(pl.col("is_candidate")).height
    total_count = out_df.height
    print(f"Wrote {total_count} rows to {output_path}")
    print(f"Candidate conversations: {candidate_count} ({candidate_count / max(total_count, 1):.2%})")


if __name__ == "__main__":
    main()
