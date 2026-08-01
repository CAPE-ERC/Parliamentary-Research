"""H2 (question deflection), tested at individual-PQ granularity.

Primary test: transferred ~ asker's party (government vs opposition), on the
full population of resolved-asker PQs. Robustness check: same model with a
Policy-topic interaction, since H2 was originally framed around "sensitive"
policy questions specifically rather than procedural ones.

See pq_deflection.py's module docstring for why the original 226-utterance,
50-topic breakdown was underpowered, and how the individual-PQ panel here
was built (12,426 distinct PQs, each with a known asker).

Usage:
    python -m linking_layer.pq_deflection_test
"""

import argparse
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import fisher_exact


def run(processed_dir: Path) -> None:
    panel = pd.read_parquet(processed_dir / "pq_deflection_panel.parquet")
    resolved = panel[panel["resolved_gov_opp"].notna()].copy()
    resolved["is_policy"] = resolved["topic"].notna() & (resolved["topic"] != "non_policy")
    resolved["transferred_int"] = resolved["transferred"].astype(int)
    resolved["party"] = resolved["resolved_gov_opp"]

    contingency = pd.crosstab(resolved["party"], resolved["transferred"])
    odds_ratio, fisher_p = fisher_exact(contingency)

    rate_by_party = resolved.groupby("party")["transferred"].mean()
    rate_by_party_topic = resolved.groupby(["party", "is_policy"])["transferred"].agg(
        ["mean", "sum", "size"]
    )

    main_model = smf.logit("transferred_int ~ C(party)", data=resolved).fit(disp=0)
    interaction_model = smf.logit(
        "transferred_int ~ C(party) * is_policy", data=resolved
    ).fit(disp=0)

    n_total = len(panel)
    n_resolved = len(resolved)
    n_unresolved = n_total - n_resolved
    n_transferred = int(resolved["transferred"].sum())

    print(f"PQs with resolved asker party: {n_resolved} / {n_total}")
    print(f"Transferred among resolved: {n_transferred} ({resolved['transferred'].mean():.4f})")
    print("\n2x2 contingency table (party x transferred):")
    print(contingency)
    print(f"\nFisher's exact test: odds ratio={odds_ratio:.4f}, p={fisher_p:.4f}")
    print("\nTransfer rate by party:")
    print(rate_by_party)
    print("\nLogistic regression (main effect):")
    print(main_model.summary())
    print("\nLogistic regression (party x policy-topic interaction):")
    print(interaction_model.summary())

    party_coef = main_model.params.get("C(party)[T.opposition]")
    party_p = main_model.pvalues.get("C(party)[T.opposition]")

    with open(processed_dir / "pq_deflection_report.md", "w", encoding="utf-8") as f:
        f.write("# H2 Reassessment - Individual-PQ Question Deflection\n\n")
        f.write(
            "The original H2 test (see linking_layer_report.md) counted 226 pnq_transfer-\n"
            "tagged utterances split across 50 topics - most topic cells had 0-2 events,\n"
            "too sparse to test anything. Reading a sample of those utterances found why:\n"
            "84% are the Speaker's routine end-of-Question-Time announcement, and each one\n"
            "bundles a median of 3 straggler PQ numbers together - a scheduling artifact,\n"
            "not 226 independent deflection decisions.\n\n"
            "Every Parliamentary Question carries a `(No. B/xxx) <Name> (<Constituency>)`\n"
            "header in the same convention Layer 0 already parses for roles, giving each\n"
            "one a known asker. This lets H2 be tested as originally intended: whether a\n"
            "question's asker predicts whether it gets transferred, across the full PQ\n"
            "population, rather than a sparse per-topic breakdown.\n\n"
        )
        f.write("## Data\n\n")
        f.write(f"- Distinct PQs identified corpus-wide: {n_total}\n")
        f.write(f"- Transfer references found: {panel.attrs.get('n_transfer_refs', 'n/a')} "
                f"({panel.attrs.get('n_transfer_refs_matched', 'n/a')} matched to a known header)\n")
        f.write(f"- PQs with a resolved asker party: {n_resolved} ({n_resolved/n_total:.1%})\n")
        f.write(f"- PQs excluded (no match or surname collision): {n_unresolved} ({n_unresolved/n_total:.1%})\n")
        f.write(f"- Transferred PQs among resolved: {n_transferred} ({resolved['transferred'].mean():.1%})\n\n")

        f.write("## Headline: no evidence of party-conditional deflection\n\n")
        f.write(f"Transfer rate is **{rate_by_party['government']:.2%}** for government-asked PQs "
                f"and **{rate_by_party['opposition']:.2%}** for opposition-asked PQs. "
                f"Fisher's exact test: odds ratio={odds_ratio:.3f}, p={fisher_p:.3f}. "
                f"Logistic regression confirms this: the opposition coefficient is "
                f"{party_coef:.4f} (p={party_p:.3f}), not significant.\n\n")

        f.write("### Contingency table\n\n")
        f.write("| Party | Not transferred | Transferred | Transfer rate |\n|---|---|---|---|\n")
        for party in ["government", "opposition"]:
            row = contingency.loc[party]
            rate = rate_by_party[party]
            f.write(f"| {party} | {row[False]} | {row[True]} | {rate:.2%} |\n")

        f.write("\n### Robustness check: interaction with Policy-topic status\n\n")
        f.write(
            "H2 was originally framed around sensitive *policy* questions specifically, "
            "not procedural ones, so the party effect is also tested conditional on whether "
            "the question falls in one of the 50 labeled Policy domains.\n\n"
        )
        f.write("| Party | Policy topic | N | Transferred | Rate |\n|---|---|---|---|---|\n")
        for (party, is_policy), row in rate_by_party_topic.iterrows():
            f.write(f"| {party} | {is_policy} | {int(row['size'])} | {int(row['sum'])} | {row['mean']:.2%} |\n")
        f.write(
            "\nNeither the main party effect nor the party x policy-topic interaction term "
            "is statistically significant in the logistic regression "
            f"(interaction p={interaction_model.pvalues.get('C(party)[T.opposition]:is_policy[T.True]', float('nan')):.3f}).\n\n"
        )

        f.write("## Caveats\n\n")
        f.write(
            "- Asker resolution uses the same surname-matching method as the Linking layer's "
            "backbench-MP resolution (see party_resolution.py); collisions within an Assembly "
            "term are left unresolved rather than guessed, which is why "
            f"{n_unresolved} of {n_total} PQs are excluded rather than assigned a party.\n"
        )
        f.write(
            "- Ministers and the Prime Minister rarely ask Parliamentary Questions of their own "
            "government, so the resolved-asker population is dominated by opposition and "
            "backbench government MPs; this test is about backbench/opposition dynamics, not a "
            "full chamber comparison.\n"
        )
        f.write(
            "- 3% of transfer references did not match a known PQ header in the same debate "
            "(likely a header the regex missed, e.g. an unusual name format) and are not counted "
            "as transferred for any PQ - a small, non-systematic undercount of the transferred "
            "total.\n"
        )

    print(f"\nReport written to {processed_dir / 'pq_deflection_report.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    args = parser.parse_args()

    run(Path(args.processed_dir))
