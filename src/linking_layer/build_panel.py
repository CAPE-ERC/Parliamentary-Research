"""Builds the analysis panel: intervention rate per (debate x topic x party)
cell, merging Layer 0 roles/party (+ the registry-resolved backbench party),
Topic layer labels, Procedural layer tags, and per-debate chair identity.

"Intervention" = a chair_ruling or interruption event (rule-tagger output,
_rule columns - the Procedural layer's own guidance prefers these over _lstm
for chair_ruling specifically, see procedural_layer_report.md) occurring
within the next WINDOW utterances after a substantive MP utterance, within
the same debate.

Usage:
    python -m linking_layer.build_panel
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW = 3
MIN_CELL_SIZE = 3


def resolve_final_party(utterances: pd.DataFrame, party_resolved: pd.DataFrame) -> pd.DataFrame:
    df = utterances.merge(
        party_resolved[["debate_id", "seq_index", "resolved_gov_opp"]],
        on=["debate_id", "seq_index"],
        how="left",
    )

    def final_party(row) -> str | None:
        if row["role"] in ("minister", "prime_minister", "deputy_prime_minister"):
            return "government"
        if row["role"] == "leader_of_opposition":
            return "opposition"
        if row["role"] == "unclassified_mp":
            return row["resolved_gov_opp"]
        return None  # chair roles / unresolved - not an "MP speaking" unit for H1

    df["speaker_party_final"] = df.apply(final_party, axis=1)
    return df


def flag_interventions(df: pd.DataFrame, procedural_tags: pd.DataFrame) -> pd.DataFrame:
    """Adds an `intervened` boolean: chair_ruling or interruption (rule-based)
    occurring in the next WINDOW rows within the same debate."""
    tags = procedural_tags[["debate_id", "seq_index", "chair_ruling_rule", "interruption_rule"]].copy()
    tags["any_event"] = (tags["chair_ruling_rule"] == 1) | (tags["interruption_rule"] == 1)

    df = df.merge(tags[["debate_id", "seq_index", "any_event"]], on=["debate_id", "seq_index"], how="left")
    df["any_event"] = df["any_event"].fillna(False)

    df = df.sort_values(["debate_id", "seq_index"]).reset_index(drop=True)
    intervened = np.zeros(len(df), dtype=bool)
    for _, group in df.groupby("debate_id", sort=False):
        idx = group.index.to_numpy()
        events = group["any_event"].to_numpy()
        n = len(events)
        for i in range(n):
            end = min(i + 1 + WINDOW, n)
            intervened[idx[i]] = events[i + 1 : end].any()
    df["intervened"] = intervened
    return df


def run(processed_dir: Path) -> None:
    utterances = pd.read_parquet(processed_dir / "utterances.parquet")
    party_resolved = pd.read_parquet(processed_dir / "speaker_party_resolved.parquet")
    topics = pd.read_parquet(processed_dir / "utterance_policy_labels_two_stage.parquet")
    procedural_tags = pd.read_parquet(processed_dir / "procedural_tags_final.parquet")
    chair_identity = pd.read_parquet(processed_dir / "chair_identity.parquet")

    print("Resolving final speaker party (Layer 0 roles + registry-resolved backbenchers)...")
    df = resolve_final_party(utterances, party_resolved)

    print("Flagging chair interventions in the following-utterance window...")
    df = flag_interventions(df, procedural_tags)

    df = df.merge(chair_identity[["debate_id", "chair_surname"]], on="debate_id", how="left")
    df = df.merge(topics[["debate_id", "seq_index", "predicted_label"]], on=["debate_id", "seq_index"], how="left")

    analysis_rows = df[
        (~df["is_stage_direction"])
        & (df["speaker_party_final"].notna())
        & (df["predicted_label"].notna())
        & (df["predicted_label"] != "non_policy")
        & (df["chair_surname"].notna())
    ].copy()
    print(f"Analysis-eligible utterances (resolved party + policy topic + known chair): {len(analysis_rows)}")

    panel = (
        analysis_rows.groupby(["debate_id", "chair_surname", "predicted_label", "speaker_party_final"])
        .agg(n_utterances=("intervened", "size"), n_intervened=("intervened", "sum"))
        .reset_index()
    )
    panel["intervention_rate"] = panel["n_intervened"] / panel["n_utterances"]
    panel = panel.rename(columns={"predicted_label": "topic", "speaker_party_final": "party"})

    panel_filtered = panel[panel["n_utterances"] >= MIN_CELL_SIZE].reset_index(drop=True)
    print(f"Panel cells: {len(panel)} total, {len(panel_filtered)} with n_utterances >= {MIN_CELL_SIZE}")

    panel.to_parquet(processed_dir / "linking_panel_full.parquet", index=False)
    panel_filtered.to_parquet(processed_dir / "linking_panel.parquet", index=False)

    print("\nParty distribution in panel (by utterance count):")
    print(analysis_rows["speaker_party_final"].value_counts())
    print("\nDistinct chairs:", panel_filtered["chair_surname"].nunique())
    print("Distinct topics represented:", panel_filtered["topic"].nunique())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    args = parser.parse_args()

    run(Path(args.processed_dir))
