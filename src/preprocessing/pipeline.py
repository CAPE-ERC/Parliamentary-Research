"""Orchestrates Layer 0: parse -> segment -> tag roles -> language ID -> write.

Usage:
    python -m preprocessing.pipeline --years 2015
    python -m preprocessing.pipeline            # full corpus
"""

import argparse
import re
import traceback
from pathlib import Path

import pandas as pd

from .language_id import detect_language
from .metadata import parse_metadata
from .pdf_parser import extract_pages
from .roles import resolve_role
from .segmenter import segment_utterances

FRONT_MATTER_MARKERS = (
    "CONTENTS",
    "THE CABINET",
    "PRINCIPAL OFFICERS",
    "NATIONAL ASSEMBLY",
    "The Assembly met",
    "The National Anthem was played",
)

FILENAME_RE = re.compile(
    r"Debate_No\.?_?(\d+[A-Za-z]?)_of_(\d{4})(?:_\((REVISED|UNREVISED)\))?", re.IGNORECASE
)


def find_body_start_page(pages: list[str]) -> int:
    """Return the index of the first page of actual debate dialogue."""
    last_marker_page = 0
    for i, page in enumerate(pages[:15]):
        if any(marker in page for marker in FRONT_MATTER_MARKERS):
            last_marker_page = i
    return min(last_marker_page + 1, len(pages) - 1)


def parse_filename(pdf_path: Path) -> dict:
    m = FILENAME_RE.search(pdf_path.name)
    return {
        "filename_debate_number": m.group(1) if m else None,
        "filename_year": m.group(2) if m else None,
        "revision_status": (m.group(3) or "").upper() if m else None,
    }


def process_debate(pdf_path: Path, year: str) -> tuple[list[dict], str | None]:
    """Returns (rows, error). rows is empty if error is set."""
    try:
        pages = extract_pages(pdf_path)
        metadata = parse_metadata(pages)
        body_start = find_body_start_page(pages)
        utterances = segment_utterances(pages, start_page=body_start)
        filename_info = parse_filename(pdf_path)

        debate_id = pdf_path.stem
        rows = []
        for u in utterances:
            role, party = (None, None)
            if u.speaker_raw and not u.is_stage_direction:
                resolved = resolve_role(u.speaker_raw, metadata)
                role, party = resolved.role, resolved.party

            rows.append(
                {
                    "debate_id": debate_id,
                    "year": year,
                    "source_file": pdf_path.name,
                    "debate_number": metadata.debate_number or filename_info["filename_debate_number"],
                    "assembly": metadata.assembly,
                    "session": metadata.session,
                    "sitting_date": metadata.sitting_date,
                    "revision_status": filename_info["revision_status"],
                    "seq_index": u.seq_index,
                    "speaker_raw": u.speaker_raw,
                    "role": role,
                    "party": party,
                    "language": None if u.is_stage_direction else detect_language(u.text),
                    "text": u.text,
                    "is_stage_direction": u.is_stage_direction,
                }
            )
        return rows, None
    except Exception as exc:  # noqa: BLE001 - one bad PDF shouldn't kill the batch
        return [], f"{pdf_path.name}: {exc}\n{traceback.format_exc(limit=3)}"


def write_quality_report(df: pd.DataFrame, failures: list[str], out_path: Path) -> None:
    dialogue = df[~df["is_stage_direction"]]
    lines = [
        "# Corpus Quality Report",
        "",
        f"- Debates processed: {df['debate_id'].nunique()}",
        f"- Debates failed to parse: {len(failures)}",
        f"- Total utterance records: {len(df)} (dialogue: {len(dialogue)}, stage directions: {len(df) - len(dialogue)})",
        "",
        "## Role resolution (dialogue utterances)",
        "",
    ]
    role_counts = dialogue["role"].value_counts(dropna=False)
    for role, count in role_counts.items():
        pct = 100 * count / len(dialogue) if len(dialogue) else 0
        lines.append(f"- {role or 'None (unresolved)'}: {count} ({pct:.1f}%)")

    lines += ["", "## Language distribution (dialogue utterances)", ""]
    lang_counts = dialogue["language"].value_counts(dropna=False)
    for lang, count in lang_counts.items():
        pct = 100 * count / len(dialogue) if len(dialogue) else 0
        lines.append(f"- {lang or 'None'}: {count} ({pct:.1f}%)")

    if failures:
        lines += ["", "## Failed files", ""]
        lines += [f"- {f.splitlines()[0]}" for f in failures]

    out_path.write_text("\n".join(lines), encoding="utf-8")


def run(raw_dir: Path, processed_dir: Path, years: list[str] | None = None) -> None:
    all_years = years or sorted(p.name for p in raw_dir.iterdir() if p.is_dir())
    (processed_dir / "utterances").mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    failures: list[str] = []

    for year in all_years:
        year_dir = raw_dir / year
        if not year_dir.exists():
            continue
        year_rows: list[dict] = []
        pdf_files = sorted(year_dir.glob("*.pdf"))
        for pdf_path in pdf_files:
            rows, error = process_debate(pdf_path, year)
            if error:
                failures.append(error)
                continue
            year_rows.extend(rows)

        if year_rows:
            year_df = pd.DataFrame(year_rows)
            year_df.to_parquet(processed_dir / "utterances" / f"{year}.parquet", index=False)
            all_rows.extend(year_rows)
            print(f"{year}: {len(pdf_files)} files, {len(year_rows)} utterances")

    if not all_rows:
        print("No utterances produced.")
        return

    combined = pd.DataFrame(all_rows)
    combined.to_parquet(processed_dir / "utterances.parquet", index=False)
    write_quality_report(combined, failures, processed_dir / "corpus_quality_report.md")
    print(f"\nTotal: {len(combined)} utterances across {combined['debate_id'].nunique()} debates.")
    print(f"Failures: {len(failures)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="*", default=None, help="e.g. --years 2015 2016")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    args = parser.parse_args()

    run(Path(args.raw_dir), Path(args.processed_dir), args.years)
