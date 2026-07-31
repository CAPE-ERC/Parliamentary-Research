"""Applies the rule-based tagger across the corpus and writes silver labels.

Usage:
    python -m procedural_layer.pipeline
"""

import argparse
from pathlib import Path

import pandas as pd

from .rules import tag_utterance

TAG_COLUMNS = ["chair_ruling", "interruption", "withdrawal_request", "so_citation", "pnq_transfer"]


def run_rules(df: pd.DataFrame) -> pd.DataFrame:
    tags = [
        tag_utterance(row.text, row.is_stage_direction, row.role)
        for row in df.itertuples(index=False)
    ]
    tags_df = pd.DataFrame(tags, columns=TAG_COLUMNS)
    out = df[["debate_id", "seq_index"]].reset_index(drop=True)
    return pd.concat([out, tags_df], axis=1)


def write_summary(tagged: pd.DataFrame, total_rows: int, out_path: Path) -> None:
    lines = ["# Procedural Layer - Rule-Based Tagging Summary", ""]
    lines.append(f"- Total utterance records scanned: {total_rows}")
    lines.append("")
    lines.append("## Tag counts")
    lines.append("")
    for col in TAG_COLUMNS:
        count = int(tagged[col].sum())
        pct = 100 * count / total_rows if total_rows else 0
        lines.append(f"- {col}: {count} ({pct:.2f}%)")

    lines.append("")
    lines.append("## Multi-label overlap (utterances with 2+ tags)")
    lines.append("")
    n_tags = tagged[TAG_COLUMNS].sum(axis=1)
    for k in range(2, n_tags.max() + 1 if len(n_tags) else 1):
        count = int((n_tags == k).sum())
        if count:
            lines.append(f"- exactly {k} tags: {count}")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def run(processed_dir: Path) -> None:
    df = pd.read_parquet(processed_dir / "utterances.parquet")
    tagged = run_rules(df)
    tagged.to_parquet(processed_dir / "procedural_tags_rules.parquet", index=False)
    write_summary(tagged, len(df), processed_dir / "procedural_rules_summary.md")

    print(f"Tagged {len(tagged)} utterances.")
    for col in TAG_COLUMNS:
        print(f"  {col}: {int(tagged[col].sum())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    args = parser.parse_args()

    run(Path(args.processed_dir))
