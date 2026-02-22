from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from gabriel_common import AGENCY_ATTRIBUTES, ensure_parent_dir, write_json

DIRECTNESS_SENSITIVE_ATTRIBUTES = [
    "personal_agency",
    "proxy_agency",
    "collective_agency",
]

CONVERGENT_COMPONENTS = {
    "personal_agency": [
        "goal_setting_count",
        "goal_navigation_count",
        "problem_decomposition_count",
        "strategic_planning_count",
        "progress_monitoring_count",
        "obstacle_management_count",
        "self_efficacy_or_confidence_count",
        "resilience_or_persistence_count",
    ],
    "proxy_agency": [
        "help_seeking_resourcefulness_count",
    ],
    "collective_agency": [
        "help_seeking_resourcefulness_count",
        "goal_navigation_count",
        "obstacle_management_count",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Gabriel agency rubric outputs with reliability and convergence checks."
    )
    parser.add_argument(
        "--ratings-wide",
        default="data/derived/gabriel/conversation_agency_ratings_wide.parquet",
        help="Baseline Gabriel wide ratings parquet.",
    )
    parser.add_argument(
        "--ratings-long",
        default="data/derived/gabriel/conversation_agency_ratings_long.parquet",
        help="Gabriel long ratings parquet for chunk-level analysis.",
    )
    parser.add_argument(
        "--features",
        default="data/derived/gabriel/conversation_agency_features.parquet",
        help="Gabriel feature parquet.",
    )
    parser.add_argument(
        "--variant-ratings-wide",
        default=None,
        help="Optional prompt-variant wide ratings parquet for robustness check.",
    )
    parser.add_argument(
        "--stripped-ratings-wide",
        default=None,
        help="Optional stripped-text ratings parquet for directness check.",
    )
    parser.add_argument(
        "--langextract-features",
        default="data/derived/langextract/conversation_agency_features.parquet",
        help="LangExtract feature parquet used for convergent validity.",
    )
    parser.add_argument(
        "--summary-output",
        default="data/derived/gabriel/rubric_validation_summary.json",
        help="JSON summary output path.",
    )
    parser.add_argument(
        "--samples-output",
        default="data/derived/gabriel/rubric_validation_samples.csv",
        help="CSV output path for high/low agency samples.",
    )
    parser.add_argument(
        "--min-reliability-corr",
        type=float,
        default=0.8,
        help="Threshold for reliability gate.",
    )
    parser.add_argument(
        "--min-robustness-corr",
        type=float,
        default=0.9,
        help="Threshold for prompt robustness gate.",
    )
    return parser.parse_args()


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0 or len(a) != len(b):
        return float("nan")
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compute_split_half_reliability(long_df: pl.DataFrame) -> dict[str, float]:
    correlations: dict[str, float] = {}
    for attribute in AGENCY_ATTRIBUTES:
        sub = long_df.filter(pl.col("attribute") == attribute)
        if sub.is_empty():
            correlations[attribute] = float("nan")
            continue

        split_df = (
            sub.with_columns(
                (pl.col("chunk_index") % 2).cast(pl.Utf8).alias("split_flag"),
            )
            .group_by(["conversation_id", "split_flag"])
            .agg(pl.col("score").mean().alias("split_mean"))
            .pivot(values="split_mean", index="conversation_id", on="split_flag")
            .rename({"0": "split_a", "1": "split_b"})
        )
        if "split_a" not in split_df.columns or "split_b" not in split_df.columns:
            correlations[attribute] = float("nan")
            continue

        valid = split_df.drop_nulls(["split_a", "split_b"])
        if valid.height < 3:
            correlations[attribute] = float("nan")
            continue
        corr = pearson_corr(
            valid["split_a"].to_numpy().astype(float),
            valid["split_b"].to_numpy().astype(float),
        )
        correlations[attribute] = corr
    return correlations


def compute_prompt_robustness(
    baseline_df: pl.DataFrame, variant_path: str | None
) -> dict[str, Any]:
    if not variant_path:
        return {"status": "not_run", "attribute_correlations": {}}
    variant_file = Path(variant_path)
    if not variant_file.exists():
        return {
            "status": "missing_variant_file",
            "variant_path": str(variant_file),
            "attribute_correlations": {},
        }
    variant_df = pl.read_parquet(variant_file)
    joined = baseline_df.join(
        variant_df.select(["conversation_id"] + AGENCY_ATTRIBUTES),
        on="conversation_id",
        how="inner",
        suffix="_variant",
    )
    correlations: dict[str, float] = {}
    for attr in AGENCY_ATTRIBUTES:
        base = joined[attr].to_numpy().astype(float)
        var = joined[f"{attr}_variant"].to_numpy().astype(float)
        correlations[attr] = pearson_corr(base, var)
    return {
        "status": "ok",
        "variant_path": str(variant_file),
        "n_overlapping_conversations": int(joined.height),
        "attribute_correlations": correlations,
    }


def compute_directness(
    baseline_df: pl.DataFrame, stripped_path: str | None
) -> dict[str, Any]:
    if not stripped_path:
        return {"status": "not_run", "mean_deltas": {}}
    stripped_file = Path(stripped_path)
    if not stripped_file.exists():
        return {
            "status": "missing_stripped_file",
            "stripped_path": str(stripped_file),
            "mean_deltas": {},
        }
    stripped_df = pl.read_parquet(stripped_file)
    joined = baseline_df.join(
        stripped_df.select(["conversation_id"] + AGENCY_ATTRIBUTES),
        on="conversation_id",
        how="inner",
        suffix="_stripped",
    )
    deltas: dict[str, float] = {}
    for attr in AGENCY_ATTRIBUTES:
        base = joined[attr].to_numpy().astype(float)
        stripped = joined[f"{attr}_stripped"].to_numpy().astype(float)
        deltas[attr] = float(np.mean(base - stripped))
    sensitive_drops = {
        k: v for k, v in deltas.items() if k in DIRECTNESS_SENSITIVE_ATTRIBUTES
    }
    return {
        "status": "ok",
        "stripped_path": str(stripped_file),
        "n_overlapping_conversations": int(joined.height),
        "mean_deltas": deltas,
        "sensitive_mean_delta": float(np.mean(list(sensitive_drops.values())))
        if sensitive_drops
        else float("nan"),
    }


def compute_convergent_validity(
    baseline_df: pl.DataFrame, langextract_path: Path
) -> dict[str, Any]:
    if not langextract_path.exists():
        return {
            "status": "langextract_missing",
            "langextract_path": str(langextract_path),
            "attribute_correlations": {},
        }
    lang_df = pl.read_parquet(langextract_path)
    valid_lang_cols = sorted(
        {
            col
            for cols in CONVERGENT_COMPONENTS.values()
            for col in cols
            if col in lang_df.columns
        }
    )
    joined = baseline_df.join(
        lang_df.select(["conversation_id"] + valid_lang_cols),
        on="conversation_id",
        how="inner",
    )
    correlations: dict[str, float] = {}
    for attr, lang_cols in CONVERGENT_COMPONENTS.items():
        available = [col for col in lang_cols if col in joined.columns]
        if not available:
            correlations[attr] = float("nan")
            continue
        proxy_target = joined.select(available).to_numpy().astype(float).mean(axis=1)
        correlations[attr] = pearson_corr(
            joined[attr].to_numpy().astype(float),
            proxy_target,
        )
    return {
        "status": "ok",
        "langextract_path": str(langextract_path),
        "n_overlapping_conversations": int(joined.height),
        "attribute_correlations": correlations,
    }


def gate_check(values: dict[str, float], threshold: float) -> dict[str, Any]:
    finite = [v for v in values.values() if np.isfinite(v)]
    if not finite:
        return {"pass": False, "mean": float("nan"), "threshold": threshold}
    mean_val = float(np.mean(finite))
    return {"pass": mean_val >= threshold, "mean": mean_val, "threshold": threshold}


def main() -> None:
    args = parse_args()
    ratings_wide_path = Path(args.ratings_wide)
    ratings_long_path = Path(args.ratings_long)
    features_path = Path(args.features)
    summary_path = Path(args.summary_output)
    samples_path = Path(args.samples_output)

    for required in [ratings_wide_path, ratings_long_path, features_path]:
        if not required.exists():
            raise FileNotFoundError(f"Required file not found: {required}")

    wide_df = pl.read_parquet(ratings_wide_path)
    long_df = pl.read_parquet(ratings_long_path)
    features_df = pl.read_parquet(features_path)

    missing_wide_columns = [c for c in AGENCY_ATTRIBUTES if c not in wide_df.columns]
    if missing_wide_columns:
        raise ValueError(f"Missing agency rating columns in wide ratings: {missing_wide_columns}")

    reliability = compute_split_half_reliability(long_df)
    robustness = compute_prompt_robustness(wide_df, args.variant_ratings_wide)
    directness = compute_directness(wide_df, args.stripped_ratings_wide)
    convergent = compute_convergent_validity(wide_df, Path(args.langextract_features))

    reliability_gate = gate_check(reliability, args.min_reliability_corr)
    robustness_gate = (
        gate_check(robustness["attribute_correlations"], args.min_robustness_corr)
        if robustness.get("status") == "ok"
        else {"pass": None, "mean": float("nan"), "threshold": args.min_robustness_corr}
    )

    missingness = {}
    score_range_check = {}
    for attr in AGENCY_ATTRIBUTES:
        nulls = wide_df[attr].null_count()
        missingness[attr] = float(nulls / max(wide_df.height, 1))
        min_val = float(wide_df[attr].min()) if wide_df[attr].min() is not None else float("nan")
        max_val = float(wide_df[attr].max()) if wide_df[attr].max() is not None else float("nan")
        score_range_check[attr] = {
            "min": min_val,
            "max": max_val,
            "in_range_0_100": bool(np.isfinite(min_val) and np.isfinite(max_val) and min_val >= 0 and max_val <= 100),
        }
    missingness_pass = all(rate < 0.02 for rate in missingness.values())
    score_range_pass = all(v["in_range_0_100"] for v in score_range_check.values())

    summary_payload: dict[str, Any] = {
        "inputs": {
            "ratings_wide": str(ratings_wide_path),
            "ratings_long": str(ratings_long_path),
            "features": str(features_path),
            "variant_ratings_wide": args.variant_ratings_wide,
            "stripped_ratings_wide": args.stripped_ratings_wide,
            "langextract_features": args.langextract_features,
        },
        "gates": {
            "reliability": reliability_gate,
            "prompt_robustness": robustness_gate,
            "missingness": {"pass": missingness_pass, "max_allowed": 0.02},
            "data_integrity": {
                "pass": bool(
                    features_df["conversation_id"].n_unique() == features_df.height
                    and score_range_pass
                ),
                "n_rows": int(features_df.height),
                "n_unique_conversation_id": int(features_df["conversation_id"].n_unique()),
                "score_range_pass": score_range_pass,
            },
            "directness": {
                "pass": None
                if directness.get("status") != "ok"
                else bool(directness.get("sensitive_mean_delta", 0.0) > 0),
                "sensitive_mean_delta": directness.get("sensitive_mean_delta"),
            },
        },
        "reliability_split_half_correlations": reliability,
        "prompt_robustness": robustness,
        "directness": directness,
        "convergent_validity": convergent,
        "missingness_rates": missingness,
        "score_ranges": score_range_check,
    }

    high_samples = features_df.sort("agency_index_raw", descending=True).head(20)
    low_samples = features_df.sort("agency_index_raw", descending=False).head(20)
    samples = pl.concat([high_samples, low_samples], how="vertical").with_columns(
        pl.when(pl.col("agency_index_raw") >= 0)
        .then(pl.lit("high"))
        .otherwise(pl.lit("low"))
        .alias("sample_group")
    )

    ensure_parent_dir(summary_path)
    write_json(summary_path, summary_payload)
    ensure_parent_dir(samples_path)
    samples.write_csv(samples_path)

    print(f"Wrote rubric validation summary: {summary_path}")
    print(f"Wrote rubric validation samples: {samples_path}")


if __name__ == "__main__":
    main()
