"""H3 (attention-cooling): does procedural conflict on a topic in sitting t
predict that topic's attention share in sitting t+1?

Per-topic Granger causality tests (does conflict_rate Granger-cause
attention_share, i.e. do conflict_rate's past values improve prediction of
attention_share beyond attention_share's own past?) plus a pooled MixedLM
panel model as a robustness cross-check, consistent with the Linking layer's
approach to H1.

Caveat, stated not hidden: Granger causality technically assumes stationary
series. A 2015-2025 span plausibly has trends/structural breaks (e.g.
COVID-era health-topic spikes). Full per-topic unit-root testing wasn't run
(50 series) - this is a limitation on interpretation, noted in the report.

Usage:
    python -m outcome_layer.granger_test
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import grangercausalitytests

MAX_LAG = 3
MIN_NONZERO_CONFLICT_SITTINGS = 10  # need real variation in conflict_rate to test anything


def per_topic_granger(series: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for topic, group in series.groupby("topic"):
        group = group.sort_values("sitting_order")
        n_conflict_sittings = (group["conflict_rate"] > 0).sum()
        if n_conflict_sittings < MIN_NONZERO_CONFLICT_SITTINGS:
            rows.append({"topic": topic, "n_sittings": len(group), "n_conflict_sittings": n_conflict_sittings,
                         "lag": None, "p_value": None, "skipped": True})
            continue

        data = group[["attention_share", "conflict_rate"]].to_numpy()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = grangercausalitytests(data, maxlag=MAX_LAG, verbose=False)
        except Exception as exc:  # noqa: BLE001
            rows.append({"topic": topic, "n_sittings": len(group), "n_conflict_sittings": n_conflict_sittings,
                         "lag": None, "p_value": None, "skipped": True, "error": str(exc)})
            continue

        for lag in range(1, MAX_LAG + 1):
            p_value = result[lag][0]["ssr_ftest"][1]
            rows.append({
                "topic": topic,
                "n_sittings": len(group),
                "n_conflict_sittings": n_conflict_sittings,
                "lag": lag,
                "p_value": p_value,
                "skipped": False,
            })
    return pd.DataFrame(rows)


def pooled_panel_model(series: pd.DataFrame) -> tuple:
    """attention_share_{t+1} ~ attention_share_t + conflict_rate_t, random
    intercept by topic - a robustness cross-check less sensitive to any
    single topic's short series than the per-topic tests alone."""
    df = series.sort_values(["topic", "sitting_order"]).copy()
    df["attention_share_next"] = df.groupby("topic")["attention_share"].shift(-1)
    df = df.dropna(subset=["attention_share_next"])

    model = smf.mixedlm(
        "attention_share_next ~ attention_share + conflict_rate",
        data=df,
        groups=df["topic"],
    )
    return model.fit(reml=False), df


def run(processed_dir: Path) -> None:
    series = pd.read_parquet(processed_dir / "outcome_series.parquet")

    print("Running per-topic Granger tests (lag 1-3)...")
    granger_results = per_topic_granger(series)
    tested = granger_results[~granger_results["skipped"]]
    lag1 = tested[tested["lag"] == 1]

    n_sig_raw = (lag1["p_value"] < 0.05).sum()
    bonferroni_alpha = 0.05 / len(lag1) if len(lag1) else float("nan")
    n_sig_bonferroni = (lag1["p_value"] < bonferroni_alpha).sum()
    n_skipped = granger_results["skipped"].sum() // (MAX_LAG if False else 1)
    n_topics_skipped = granger_results[granger_results["skipped"]]["topic"].nunique()

    print(f"Lag-1: {n_sig_raw} of {len(lag1)} topics significant at raw p<0.05 "
          f"({n_sig_bonferroni} survive Bonferroni, alpha={bonferroni_alpha:.5f})")
    print(f"Topics skipped (too little conflict variation): {n_topics_skipped}")

    print("\nFitting pooled panel model (robustness cross-check)...")
    pooled_result, panel_df = pooled_panel_model(series)
    print(pooled_result.summary())

    with open(processed_dir / "outcome_layer_report.md", "w", encoding="utf-8") as f:
        f.write("# Outcome Layer - Time-Series / Granger-Style Report (H3)\n\n")
        f.write(f"- Topics modeled: {series['topic'].nunique()}\n")
        f.write(f"- Sittings (chronological): {series['sitting_order'].nunique()}\n")
        f.write(f"- Topics with enough conflict variation to test: {len(lag1)} of {series['topic'].nunique()} "
                f"(need >= {MIN_NONZERO_CONFLICT_SITTINGS} sittings with conflict_rate > 0)\n\n")

        f.write("## Headline: lag-1 Granger results (does conflict_t predict attention_{t+1}?)\n\n")
        f.write(f"**{n_sig_raw} of {len(lag1)} topics significant at raw p<0.05** "
                f"({n_sig_bonferroni} survive Bonferroni correction, alpha={bonferroni_alpha:.5f}).\n\n")
        f.write("| Topic | N sittings | N conflict sittings | p-value (lag 1) |\n|---|---|---|---|\n")
        for _, row in lag1.sort_values("p_value").iterrows():
            f.write(f"| {row['topic']} | {row['n_sittings']} | {row['n_conflict_sittings']} | {row['p_value']:.4f} |\n")

        f.write("\n## Lag 2-3 results (robustness)\n\n")
        f.write("| Topic | Lag | p-value |\n|---|---|---|\n")
        for _, row in tested[tested["lag"] > 1].sort_values(["topic", "lag"]).iterrows():
            f.write(f"| {row['topic']} | {row['lag']} | {row['p_value']:.4f} |\n")

        f.write("\n## Pooled panel model (robustness cross-check)\n\n")
        f.write(f"`attention_share_(t+1) ~ attention_share_t + conflict_rate_t`, random intercept by topic, "
                f"n={len(panel_df)}, converged={pooled_result.converged}\n\n")
        f.write("```\n" + str(pooled_result.summary()) + "\n```\n")

        f.write("\n## Caveats\n\n")
        f.write(
            "- **Stationarity not tested.** Granger causality assumes stationary series; a 2015-2025 "
            "span plausibly has trends/structural breaks (e.g. COVID-era health-topic spikes). Full "
            "per-topic ADF unit-root testing wasn't run given scope (50 series) - treat results as "
            "suggestive, not confirmatory, without that check.\n"
        )
        f.write(
            f"- {n_topics_skipped} topics had fewer than {MIN_NONZERO_CONFLICT_SITTINGS} sittings with "
            "nonzero conflict_rate and were skipped - too little variation in the predictor to test "
            "meaningfully, not evidence of no effect for those topics specifically.\n"
        )
        f.write(
            "- With 47-50 near-independent per-topic tests, some raw p<0.05 results are expected by "
            "chance alone (~2-2.5 at a 5% rate) - the Bonferroni-adjusted count is the more trustworthy "
            "summary statistic.\n"
        )

    print(f"\nReport written to {processed_dir / 'outcome_layer_report.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    args = parser.parse_args()

    run(Path(args.processed_dir))
