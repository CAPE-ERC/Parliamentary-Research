"""H1: intervention_rate ~ topic * party, random intercept by chair (MixedLM).
H2 (secondary, simpler): pnq_transfer rate by topic x party - not forced into
the same panel/MixedLM structure, since it's a much rarer, more narrowly
formulaic event (see procedural_layer_report.md).

Usage:
    python -m linking_layer.regression
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

MIN_TOPIC_CELLS = 10  # drop topics too sparse in the panel to estimate reliably


def fit_h1(panel: pd.DataFrame) -> tuple:
    topic_counts = panel["topic"].value_counts()
    keep_topics = topic_counts[topic_counts >= MIN_TOPIC_CELLS].index
    dropped = sorted(set(panel["topic"]) - set(keep_topics))
    model_df = panel[panel["topic"].isin(keep_topics)].copy()

    model = smf.mixedlm(
        "intervention_rate ~ C(topic) * C(party)",
        data=model_df,
        groups=model_df["chair_surname"],
    )
    result = model.fit(reml=False)
    return result, model_df, dropped


def h2_pnq_deflection(processed_dir: Path, panel_full: pd.DataFrame) -> pd.DataFrame:
    procedural_tags = pd.read_parquet(processed_dir / "procedural_tags_final.parquet")
    utterances = pd.read_parquet(processed_dir / "utterances.parquet")
    topics = pd.read_parquet(processed_dir / "utterance_policy_labels_two_stage.parquet")

    df = utterances.merge(procedural_tags[["debate_id", "seq_index", "pnq_transfer_rule"]], on=["debate_id", "seq_index"])
    df = df.merge(topics[["debate_id", "seq_index", "predicted_label"]], on=["debate_id", "seq_index"])
    df = df[(~df["is_stage_direction"]) & (df["predicted_label"] != "non_policy")]

    by_topic = (
        df.groupby("predicted_label")
        .agg(n=("pnq_transfer_rule", "size"), n_transfer=("pnq_transfer_rule", "sum"))
        .reset_index()
    )
    by_topic["transfer_rate"] = by_topic["n_transfer"] / by_topic["n"]
    return by_topic.sort_values("transfer_rate", ascending=False)


def run(processed_dir: Path) -> None:
    panel = pd.read_parquet(processed_dir / "linking_panel.parquet")
    print(f"Panel: {len(panel)} rows, {panel['topic'].nunique()} topics, "
          f"{panel['chair_surname'].nunique()} chairs")

    print("\nFitting H1: intervention_rate ~ C(topic) * C(party), random intercept by chair...")
    result, model_df, dropped_topics = fit_h1(panel)
    print(result.summary())

    # Party×topic interaction terms only (the H1-relevant coefficients)
    interaction_terms = result.params[result.params.index.str.contains(":")]
    interaction_pvalues = result.pvalues[result.pvalues.index.str.contains(":")]

    print("\nComputing H2 (PNQ deflection by topic)...")
    h2_table = h2_pnq_deflection(processed_dir, panel)

    with open(processed_dir / "linking_layer_report.md", "w", encoding="utf-8") as f:
        f.write("# Linking Layer - Regression Report\n\n")
        f.write(f"- Panel rows used: {len(model_df)} (of {len(panel)} total; "
                f"{len(dropped_topics)} topics dropped for having fewer than {MIN_TOPIC_CELLS} panel cells)\n")
        f.write(f"- Distinct topics modeled: {model_df['topic'].nunique()}\n")
        f.write(f"- Distinct chairs (random effect groups): {model_df['chair_surname'].nunique()}\n")
        f.write(f"- Converged: {result.converged}\n\n")

        f.write("## H1: topic x party interaction terms (asymmetric enforcement)\n\n")
        f.write("| Term | Coefficient | p-value |\n|---|---|---|\n")
        for term in interaction_terms.index:
            f.write(f"| {term} | {interaction_terms[term]:.4f} | {interaction_pvalues[term]:.4f} |\n")
        n_sig = (interaction_pvalues < 0.05).sum()
        f.write(f"\n{n_sig} of {len(interaction_terms)} topic x opposition interaction terms "
                f"significant at p<0.05.\n\n")

        f.write("## H2: PNQ deflection rate by topic (secondary analysis)\n\n")
        f.write("| Topic | N | N transferred | Transfer rate |\n|---|---|---|---|\n")
        for _, row in h2_table.head(15).iterrows():
            f.write(f"| {row['predicted_label']} | {row['n']} | {row['n_transfer']} | {row['transfer_rate']:.4f} |\n")

        f.write("\n\nFull model summary:\n\n```\n" + str(result.summary()) + "\n```\n")

    print(f"\n{n_sig} of {len(interaction_terms)} topic x party interaction terms significant at p<0.05")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    args = parser.parse_args()

    run(Path(args.processed_dir))
