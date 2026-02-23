from __future__ import annotations

import argparse
import random
from pathlib import Path

import langextract as lx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an HTML visualization for one random LangExtract annotated document."
    )
    parser.add_argument(
        "--input-jsonl",
        default="data/derived/langextract/langextract_annotations.jsonl",
        help="Path to LangExtract annotations JSONL.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="analysis/langextract/06_random_langextract_sample.jsonl",
        help="Path to save the single sampled annotated document JSONL.",
    )
    parser.add_argument(
        "--output-html",
        default="analysis/langextract/06_random_langextract_visualization.html",
        help="Path to save the generated HTML visualization.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility.",
    )
    parser.add_argument(
        "--allow-empty-extractions",
        action="store_true",
        help="Allow sampling documents with zero extractions.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    input_jsonl = Path(args.input_jsonl)
    output_jsonl = Path(args.output_jsonl)
    output_html = Path(args.output_html)

    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")

    docs = list(lx.io.load_annotated_documents_jsonl(input_jsonl, show_progress=False))
    if not docs:
        raise ValueError(f"No annotated documents found in: {input_jsonl}")

    if args.allow_empty_extractions:
        pool = docs
    else:
        pool = [doc for doc in docs if (doc.extractions or [])]
        if not pool:
            raise ValueError(
                "No documents with extractions were found. "
                "Use --allow-empty-extractions to sample anyway."
            )

    rng = random.Random(args.seed)
    sampled_doc = rng.choice(pool)

    ensure_parent(output_jsonl)
    lx.io.save_annotated_documents(
        [sampled_doc],
        output_name=output_jsonl.name,
        output_dir=output_jsonl.parent,
        show_progress=False,
    )

    html_content = lx.visualize(str(output_jsonl))

    ensure_parent(output_html)
    with output_html.open("w", encoding="utf-8") as f:
        if hasattr(html_content, "data"):
            f.write(html_content.data)
        else:
            f.write(str(html_content))

    extraction_count = len(sampled_doc.extractions or [])
    print(f"Sampled conversation_id: {sampled_doc.document_id}")
    print(f"Sampled extraction count: {extraction_count}")
    print(f"Wrote sampled JSONL: {output_jsonl}")
    print(f"Wrote visualization HTML: {output_html}")


if __name__ == "__main__":
    main()
