"""Figures for the revised manuscript (H1-H3 interval plots).

Reads the CSV outputs of the H1, H2 and H3 reanalyses and writes PNGs to
docs/figures/. Each figure is a single-series interval plot: point estimate
with 95% CI, a neutral reference line at no effect, and dashed equivalence
bounds where the paper uses them.

Usage:
    python -m reporting.paper_figures
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

INK = "#1F2933"
MUTED = "#6B7280"
GRID = "#E5E7EB"
SERIES = "#1F3A5F"  # navy used in the manuscript's existing figures

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
})


def interval_plot(ax, labels, est, lo, hi, ref=0.0, bounds=None, log=False, bold_first=True):
    y = list(range(len(labels)))[::-1]
    ax.axvline(ref, color=MUTED, lw=1, zorder=1)
    if bounds:
        for b in bounds:
            ax.axvline(b, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.hlines(y, lo, hi, color=SERIES, lw=2, zorder=2)
    ax.scatter(est, y, s=36, color=SERIES, edgecolor="white", linewidth=1.5, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    if bold_first:
        ax.get_yticklabels()[0].set_fontweight("bold")
    if log:
        ax.set_xscale("log")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)


def h1_figure(processed: Path, out: Path) -> None:
    g = pd.read_csv(processed / "h1_sensitivity.csv").set_index("specification")
    rows = [
        ("Primary: any event", "Primary: any event, window 3"),
        ("Chair rulings only", "Chair rulings only, window 3"),
        ("Interruptions only", "Interruptions only, window 3"),
        ("Backbench members only", "Backbench members only"),
        ("Backbench, chair rulings only", "Backbench only, chair rulings only"),
        ("Low-confidence topics excluded", "Excluding low-confidence topic labels"),
        ("Sixth Assembly", "Sixth Assembly: any event"),
        ("Seventh Assembly", "Seventh Assembly: any event"),
        ("Eighth Assembly", "Eighth Assembly: any event"),
    ]
    d = g.loc[[k for _, k in rows]]
    fig, ax = plt.subplots(figsize=(6.3, 3.6), dpi=200)
    interval_plot(ax, [l for l, _ in rows], d.AME_pp, d.AME_lo95_pp, d.AME_hi95_pp, bounds=[-5, 5])
    ax.set_xlabel("Opposition minus Government intervention probability\n"
                  "(percentage points, 95% CI; dashed lines = +/-5 pp)")
    ax.set_title("Figure 5. H1: party difference in intervention", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / "figure5_h1_party_effect.png")
    plt.close(fig)


def h2_figure(processed: Path, out: Path) -> None:
    party = pd.read_csv(processed / "h2_party_models.csv").set_index("specification")
    sens = pd.read_csv(processed / "h2_sensitive_contrasts.csv")
    rows = [
        ("Primary: adjusted, withdrawn excluded", party.loc["Primary: adjusted (term + section + addressee FE), withdrawn excluded"]),
        ("Unadjusted, withdrawn excluded", party.loc["Unadjusted, withdrawn PQs excluded"]),
        ("Substantive transfers only", party.loc["Adjusted, substantive transfers only (acting-minister cover excluded)"]),
        ("Oral questions", party.loc["Adjusted, oral questions only"]),
        ("Written questions", party.loc["Adjusted, written questions only"]),
    ]
    for definition, label in [("Core accountability lexicon", "Sensitive: core lexicon"),
                              ("Extended accountability lexicon", "Sensitive: extended lexicon"),
                              ("Pre-specified sensitive topics", "Sensitive: topic set")]:
        rows.append((label, sens[(sens.definition == definition) & (sens.subset == "sensitive")].iloc[0]))
    fig, ax = plt.subplots(figsize=(6.3, 3.4), dpi=200)
    interval_plot(ax, [l for l, _ in rows], [r.OR for _, r in rows], [r.OR_lo95 for _, r in rows],
                  [r.OR_hi95 for _, r in rows], ref=1.0, log=True)
    ax.set_xticks([0.5, 1, 2, 4, 8])
    ax.set_xticklabels(["0.5", "1", "2", "4", "8"])
    ax.set_xlabel("Odds ratio, Opposition vs Government asker\n(log scale, 95% CI)")
    ax.set_title("Figure 6. H2: PQ transfer by asker bloc", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / "figure6_h2_transfer_or.png")
    plt.close(fig)


def h3_figure(processed: Path, out: Path) -> None:
    pooled = pd.read_csv(processed / "h3_pooled_models.csv").iloc[0]
    topics = pd.read_csv(processed / "h3_per_topic.csv")
    mean_by_topic = topics.beta_per_sd / (topics.pct_of_mean / 100)
    topics = topics.assign(pct_lo=100 * topics.lo95 / mean_by_topic, pct_hi=100 * topics.hi95 / mean_by_topic)
    topics = topics.sort_values("pct_of_mean")
    labels = ["Pooled (topic + sitting FE)"] + [t if len(t) <= 46 else t[:44] + "..." for t in topics.topic]
    est = [pooled.pct_of_mean] + topics.pct_of_mean.tolist()
    lo = [pooled.pct_lo95] + topics.pct_lo.tolist()
    hi = [pooled.pct_hi95] + topics.pct_hi.tolist()
    fig, ax = plt.subplots(figsize=(6.3, 8.2), dpi=200)
    interval_plot(ax, labels, est, lo, hi, bounds=[-10, 10])
    ax.set_xlim(-120, 160)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xlabel("Effect of +1 SD conflict on next-sitting attention\n"
                  "(% of topic mean, 95% CI; dashed lines = +/-10%)")
    ax.set_title("Figure 7. H3: conflict and next-sitting attention", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / "figure7_h3_conflict_effect.png")
    plt.close(fig)


def run(processed: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    h1_figure(processed, out)
    h2_figure(processed, out)
    h3_figure(processed, out)
    print(f"Wrote figures to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--out-dir", default="docs/figures")
    args = parser.parse_args()
    run(Path(args.processed_dir), Path(args.out_dir))
