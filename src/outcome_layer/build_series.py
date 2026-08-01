"""Builds one topic-attention-share / procedural-conflict time series per
policy topic, indexed by chronological sitting order across all sittings
(not just the ones a topic happened to come up in) - see outcome_layer_report.md
for why: H3's "sittings that follow" reads most faithfully as the actual next
parliamentary sitting, with attention_share = 0 on days a topic isn't raised
at all, rather than skipping to the topic's next appearance.

Conflict measure reuses the Linking layer's windowed "intervened" definition
(chair_ruling/interruption within 3 utterances of a substantive turn) for
consistency across H1/H2/H3.

Usage:
    python -m outcome_layer.build_series
"""

import argparse
from pathlib import Path

import pandas as pd

from linking_layer.build_panel import flag_interventions
from linking_layer.party_resolution import parse_sitting_date


def build_chronological_debates(utterances: pd.DataFrame) -> pd.DataFrame:
    debates = utterances[["debate_id", "sitting_date"]].drop_duplicates()
    debates["parsed_date"] = debates["sitting_date"].apply(parse_sitting_date)
    dropped = debates[debates["parsed_date"].isna()]
    debates = debates.dropna(subset=["parsed_date"]).sort_values("parsed_date").reset_index(drop=True)
    debates["sitting_order"] = range(len(debates))
    return debates, dropped


def build_series(processed_dir: Path) -> pd.DataFrame:
    utterances = pd.read_parquet(processed_dir / "utterances.parquet")
    topics = pd.read_parquet(processed_dir / "utterance_policy_labels_two_stage.parquet")
    procedural_tags = pd.read_parquet(processed_dir / "procedural_tags_final.parquet")

    debates, dropped = build_chronological_debates(utterances)
    print(f"Chronologically ordered debates: {len(debates)} (dropped {len(dropped)} unparseable dates)")

    df = utterances[~utterances["is_stage_direction"]].merge(
        topics[["debate_id", "seq_index", "predicted_label"]], on=["debate_id", "seq_index"]
    )
    df = flag_interventions(df, procedural_tags)
    df = df.merge(debates[["debate_id", "sitting_order", "parsed_date"]], on="debate_id", how="inner")

    totals = df.groupby("debate_id").size().rename("total_utterances")

    policy_topics = sorted(t for t in topics["predicted_label"].unique() if t != "non_policy")
    topic_df = df[df["predicted_label"] != "non_policy"]
    per_topic_debate = (
        topic_df.groupby(["predicted_label", "debate_id"])
        .agg(n_topic=("intervened", "size"), n_intervened=("intervened", "sum"))
        .reset_index()
        .rename(columns={"predicted_label": "topic"})
    )

    grid = pd.MultiIndex.from_product(
        [policy_topics, debates["debate_id"]], names=["topic", "debate_id"]
    ).to_frame(index=False)
    grid = grid.merge(per_topic_debate, on=["topic", "debate_id"], how="left")
    grid["n_topic"] = grid["n_topic"].fillna(0)
    grid["n_intervened"] = grid["n_intervened"].fillna(0)

    grid = grid.merge(debates[["debate_id", "sitting_order", "parsed_date"]], on="debate_id", how="left")
    grid = grid.merge(totals, on="debate_id", how="left")

    grid["attention_share"] = grid["n_topic"] / grid["total_utterances"]
    grid["conflict_rate"] = (grid["n_intervened"] / grid["n_topic"]).fillna(0.0)
    grid = grid.rename(columns={"parsed_date": "sitting_date"})

    return grid.sort_values(["topic", "sitting_order"]).reset_index(drop=True)[
        ["topic", "debate_id", "sitting_order", "sitting_date", "attention_share", "conflict_rate"]
    ]


def run(processed_dir: Path) -> None:
    series = build_series(processed_dir)
    series.to_parquet(processed_dir / "outcome_series.parquet", index=False)

    n_topics = series["topic"].nunique()
    n_sittings = series["sitting_order"].nunique()
    print(f"\nBuilt {len(series)} rows: {n_topics} topics x {n_sittings} sittings")
    print("\nAttention share summary:")
    print(series["attention_share"].describe())
    print("\nConflict rate summary (non-zero only):")
    print(series[series["conflict_rate"] > 0]["conflict_rate"].describe())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    args = parser.parse_args()

    run(Path(args.processed_dir))
