"""Extracts the presiding Speaker's surname per debate (for the mixed-effects
model's chair random effect), by re-parsing just each PDF's front matter
(pages 1-8, via the existing preprocessing.metadata module) - not the full,
much more expensive utterance segmentation.

Usage:
    python -m linking_layer.chair_identity
"""

import argparse
import re
from pathlib import Path

import pandas as pd

from preprocessing.metadata import parse_metadata
from preprocessing.pdf_parser import extract_pages


def clean_surname(raw: str | None) -> str | None:
    """metadata.parse_officers() extraction isn't uniform across the corpus's
    11 years of PDF layouts - some entries come through as a full "Hon. Mrs
    Firstname Surname" blob (occasionally with a stray embedded newline/page
    number) rather than a clean surname like parse_officers gets right most
    of the time. Only a stable grouping key is needed here (not a display
    name), so reduce to the last real word."""
    if not raw:
        return None
    text = raw.replace("\n", " ")
    text = re.sub(r"\b\d+\b", " ", text)  # strip stray page-number digits
    text = re.sub(r"^(hon\.?\s+)?(mrs?\.?\s+|ms\.?\s+|dr\.?\s+)?", "", text.strip(), flags=re.IGNORECASE)
    tokens = [t for t in text.split() if t]
    return tokens[-1].lower().strip(".") if tokens else None


def run(raw_dir: Path, processed_dir: Path) -> None:
    debates = pd.read_parquet(processed_dir / "utterances.parquet")[
        ["debate_id", "source_file", "year"]
    ].drop_duplicates()

    rows = []
    failures = []
    for row in debates.itertuples(index=False):
        pdf_path = raw_dir / row.year / row.source_file
        if not pdf_path.exists():
            failures.append(row.source_file)
            continue
        try:
            pages = extract_pages(pdf_path)[:8]
            metadata = parse_metadata(pages)
            rows.append(
                {
                    "debate_id": row.debate_id,
                    "chair_surname": clean_surname(metadata.officers.get("speaker")),
                    "deputy_speaker_surname": clean_surname(metadata.officers.get("deputy_speaker")),
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{row.source_file}: {exc}")

    out = pd.DataFrame(rows)
    out.to_parquet(processed_dir / "chair_identity.parquet", index=False)

    print(f"Extracted chair identity for {len(out)} debates ({out['chair_surname'].notna().sum()} resolved).")
    print(f"Distinct chair surnames: {out['chair_surname'].nunique()}")
    print(out["chair_surname"].value_counts(dropna=False).head(10))
    if failures:
        print(f"\nFailures ({len(failures)}): {failures[:5]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    args = parser.parse_args()

    run(Path(args.raw_dir), Path(args.processed_dir))
