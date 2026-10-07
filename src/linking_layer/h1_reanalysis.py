"""H1 reanalysis (asymmetric enforcement), addressing the external review.

Rebuilds the H1 data at the utterance level so that the observation unit,
numerator, denominator and dependence structure are explicit, then reports:

  A. Unit definition, group sizes and raw intervention rates.
  B. Replication of the original specification (cell-level MixedLM, chair
     random intercept) with the party main effect, the overall marginal party
     contrast, a joint Wald test of the topic x party interactions, chair
     variance estimate, convergence, and the same model with chair FIXED
     effects.
  C. Primary revised specification: utterance-level binomial (logit) model
     of intervention, party + topic FE + chair FE + log utterance length
     (exposure), SEs clustered by sitting. Reports OR and average marginal
     effect (percentage points) with 95% CIs, an equivalence (TOST) test
     against a pre-stated smallest effect size of interest, and the minimum
     detectable effect. SEs are two-way clustered by sitting and speaker.
  D. Topic-specific party contrasts (OR with 95% CI, Bonferroni and BH) and
     the joint interaction test under the revised specification.
  E. Sensitivity grid: chair-rulings-only outcome, next-utterance window,
     backbench-only comparison, RPO excluded, Duval chair excluded,
     low-confidence topic labels excluded, sitting fixed effects, clustering
     by sitting or speaker alone, and per-Assembly-term estimates split by
     event type.

Usage:
    python -m linking_layer.h1_reanalysis
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

from linking_layer.build_panel import MIN_CELL_SIZE, WINDOW, resolve_final_party
from linking_layer.regression import MIN_TOPIC_CELLS

# Equivalence bounds. Hjorth (2016) reports a ~5 percentage-point copartisan
# difference, used as the benchmark bound; the strict bound (2 points, ~10% of
# the ~20% baseline intervention rate) asks whether even a small asymmetry
# can be ruled out.
SESOI_PP_STRICT = 0.02
SESOI_PP_BENCHMARK = 0.05
SESOI_OR = 1.11  # symmetric on the log-odds scale: [1/1.11, 1.11] ~ +/-10% odds

# Utterances are repeated within sittings and within members, so SEs are
# two-way clustered by default.
TWO_WAY = ("sitting", "speaker_id")

# Topic-specific contrasts are only estimated where both blocs have enough
# utterances and events for a stable logit estimate.
MIN_TOPIC_UTTS_PER_PARTY = 30
MIN_TOPIC_EVENTS_PER_PARTY = 5

# Permutation test of the joint interaction null: party labels are shuffled
# across speaker party-spells (a member in one bloc), preserving the
# within-member dependence that the LR test ignores.
N_PERMUTATIONS = 500
PERMUTATION_SEED = 20261007

LOW_CONFIDENCE_THRESHOLD = 0.9  # roughly the bottom quartile of classifier confidence

MINISTERIAL_ROLES = ("minister", "prime_minister", "deputy_prime_minister")


# --------------------------------------------------------------------------
# Data construction
# --------------------------------------------------------------------------

def flag_window(df: pd.DataFrame, event_col: str, window: int) -> pd.Series:
    """True where `event_col` fires in any of the next `window` rows of the
    same debate. `df` must be sorted by (debate_id, seq_index)."""
    grouped = df.groupby("debate_id", sort=False)[event_col]
    flagged = pd.Series(False, index=df.index)
    for k in range(1, window + 1):
        flagged |= grouped.shift(-k, fill_value=False).astype(bool)
    return flagged


def speaker_key(speaker_raw: str | None) -> str | None:
    if speaker_raw is None or pd.isna(speaker_raw):
        return None
    return re.sub(r"\s+", " ", speaker_raw).strip().lower()


def build_utterance_frame(processed_dir: Path) -> pd.DataFrame:
    utterances = pd.read_parquet(
        processed_dir / "utterances.parquet",
        columns=["debate_id", "seq_index", "assembly", "speaker_raw", "role", "text", "is_stage_direction"],
    )
    party_resolved = pd.read_parquet(processed_dir / "speaker_party_resolved.parquet")
    topics = pd.read_parquet(processed_dir / "utterance_policy_labels_two_stage.parquet")
    tags = pd.read_parquet(processed_dir / "procedural_tags_final.parquet")
    chairs = pd.read_parquet(processed_dir / "chair_identity.parquet")

    df = resolve_final_party(utterances, party_resolved)
    df = df.merge(party_resolved[["debate_id", "seq_index", "resolved_party"]], on=["debate_id", "seq_index"], how="left")
    df = df.merge(
        tags[["debate_id", "seq_index", "chair_ruling_rule", "interruption_rule"]],
        on=["debate_id", "seq_index"],
        how="left",
    )
    df["ruling_event"] = df["chair_ruling_rule"].fillna(0).astype(bool)
    df["interruption_event"] = df["interruption_rule"].fillna(0).astype(bool)
    df["any_event"] = df["ruling_event"] | df["interruption_event"]

    df = df.sort_values(["debate_id", "seq_index"]).reset_index(drop=True)
    df["y_any_w3"] = flag_window(df, "any_event", WINDOW)  # original H1 definition
    df["y_ruling_w3"] = flag_window(df, "ruling_event", WINDOW)
    df["y_interruption_w3"] = flag_window(df, "interruption_event", WINDOW)
    df["y_any_w1"] = flag_window(df, "any_event", 1)
    df["y_ruling_w1"] = flag_window(df, "ruling_event", 1)

    df = df.merge(chairs[["debate_id", "chair_surname"]], on="debate_id", how="left")
    df = df.merge(topics, on=["debate_id", "seq_index"], how="left")

    eligible = df[
        (~df["is_stage_direction"])
        & df["speaker_party_final"].notna()
        & df["predicted_label"].notna()
        & (df["predicted_label"] != "non_policy")
        & df["chair_surname"].notna()
    ].copy()

    eligible["opp"] = (eligible["speaker_party_final"] == "opposition").astype(int)
    eligible["role_group"] = np.where(
        eligible["role"].isin(MINISTERIAL_ROLES),
        "minister",
        np.where(eligible["role"] == "leader_of_opposition", "leader_of_opposition", "backbench"),
    )
    eligible["log_words"] = np.log1p(eligible["text"].fillna("").str.split().str.len())
    eligible["speaker_id"] = eligible["speaker_raw"].map(speaker_key).fillna("unknown")
    eligible["topic"] = eligible["predicted_label"]
    eligible["chair"] = eligible["chair_surname"]
    eligible["term"] = eligible["assembly"]
    eligible["is_rpo"] = eligible["resolved_party"].eq("RPO")
    eligible["sitting"] = eligible["debate_id"]

    keep = [
        "sitting", "seq_index", "term", "chair", "topic", "predicted_confidence", "opp", "role_group",
        "speaker_id", "is_rpo", "log_words", "y_any_w3", "y_ruling_w3", "y_interruption_w3",
        "y_any_w1", "y_ruling_w1",
    ]
    return eligible[keep].reset_index(drop=True)


def build_cell_panel(utts: pd.DataFrame, outcome: str = "y_any_w3") -> pd.DataFrame:
    """Reproduces build_panel.py's sitting x chair x topic x party cells."""
    panel = (
        utts.groupby(["sitting", "chair", "topic", "opp"])
        .agg(n_utterances=(outcome, "size"), n_intervened=(outcome, "sum"))
        .reset_index()
    )
    panel["intervention_rate"] = panel["n_intervened"] / panel["n_utterances"]
    panel = panel[panel["n_utterances"] >= MIN_CELL_SIZE].reset_index(drop=True)
    topic_counts = panel["topic"].value_counts()
    return panel[panel["topic"].isin(topic_counts[topic_counts >= MIN_TOPIC_CELLS].index)].reset_index(drop=True)


# --------------------------------------------------------------------------
# Estimation helpers
# --------------------------------------------------------------------------

def fit_logit(data: pd.DataFrame, formula: str, cluster: str | tuple[str, ...] = TWO_WAY):
    data = data.reset_index(drop=True)
    clusters = (cluster,) if isinstance(cluster, str) else cluster
    groups = np.column_stack([pd.factorize(data[c])[0] for c in clusters])
    if groups.shape[1] == 1:
        groups = groups[:, 0]
    model = smf.glm(formula, data=data, family=sm.families.Binomial())
    return model.fit(cov_type="cluster", cov_kwds={"groups": groups}), data


def average_marginal_effect(result, data: pd.DataFrame, var: str = "opp") -> tuple[float, float]:
    """AME of switching `var` 0 -> 1 for every row, with delta-method SE
    (uses the result's clustered covariance)."""
    design_info = result.model.data.design_info
    x1 = np.asarray(patsy.build_design_matrices([design_info], data.assign(**{var: 1}))[0])
    x0 = np.asarray(patsy.build_design_matrices([design_info], data.assign(**{var: 0}))[0])
    beta = result.params.to_numpy()
    p1 = 1 / (1 + np.exp(-x1 @ beta))
    p0 = 1 / (1 + np.exp(-x0 @ beta))
    ame = float(np.mean(p1 - p0))
    grad = (x1 * (p1 * (1 - p1))[:, None] - x0 * (p0 * (1 - p0))[:, None]).mean(axis=0)
    se = float(np.sqrt(grad @ result.cov_params().to_numpy() @ grad))
    return ame, se


def tost_p(estimate: float, se: float, lower: float, upper: float) -> float:
    """Two one-sided tests (normal approximation): p < 0.05 means the
    estimate is statistically within (lower, upper)."""
    p_lower = 1 - stats.norm.cdf((estimate - lower) / se)
    p_upper = stats.norm.cdf((estimate - upper) / se)
    return float(max(p_lower, p_upper))


def party_summary(label: str, result, data: pd.DataFrame, with_ame: bool = True) -> dict:
    b = result.params["opp"]
    se = result.bse["opp"]
    z95 = stats.norm.ppf(0.975)
    row = {
        "specification": label,
        "n_utterances": int(result.nobs),
        "n_opposition": int(data["opp"].sum()),
        "OR": np.exp(b),
        "OR_lo95": np.exp(b - z95 * se),
        "OR_hi95": np.exp(b + z95 * se),
        "p": result.pvalues["opp"],
        "tost_p_OR": tost_p(b, se, -np.log(SESOI_OR), np.log(SESOI_OR)),
    }
    if with_ame:
        ame, ame_se = average_marginal_effect(result, data)
        row.update(
            AME_pp=100 * ame,
            AME_lo95_pp=100 * (ame - z95 * ame_se),
            AME_hi95_pp=100 * (ame + z95 * ame_se),
            MDE80_pp=100 * (stats.norm.ppf(0.975) + stats.norm.ppf(0.80)) * ame_se,
            tost_p_pp2=tost_p(ame, ame_se, -SESOI_PP_STRICT, SESOI_PP_STRICT),
            tost_p_pp5=tost_p(ame, ame_se, -SESOI_PP_BENCHMARK, SESOI_PP_BENCHMARK),
        )
    return row


def joint_interaction_test(result) -> tuple[float, int, float]:
    names = list(result.params.index)
    idx = [i for i, n in enumerate(names) if ":" in n and "opp" in n]
    restriction = np.zeros((len(idx), len(names)))
    for r, i in enumerate(idx):
        restriction[r, i] = 1
    wald = result.wald_test(restriction, scalar=True)
    return float(wald.statistic), len(idx), float(wald.pvalue)


# --------------------------------------------------------------------------
# Analyses
# --------------------------------------------------------------------------

def replicate_original(panel: pd.DataFrame) -> dict:
    """Original Eq. (1): cell-level MixedLM with chair random intercept, plus
    the same model with chair fixed effects (OLS, SEs clustered by sitting)."""
    out = {}
    ml = smf.mixedlm("intervention_rate ~ C(topic) * opp", data=panel, groups=panel["chair"]).fit(reml=False)
    names = list(ml.fe_params.index)
    beta = ml.fe_params.to_numpy()
    cov = ml.cov_params().iloc[: len(names), : len(names)].to_numpy()
    interaction_names = [n for n in names if ":opp" in n]

    # Marginal party contrast: average of topic-specific Opp-Gov differences,
    # weighted by each topic's share of panel cells.
    weights = panel["topic"].value_counts(normalize=True)
    contrast = np.zeros(len(names))
    contrast[names.index("opp")] = 1.0
    for n in interaction_names:
        topic = re.search(r"C\(topic\)\[T\.(.*)\]:opp", n).group(1)
        contrast[names.index(n)] = weights.get(topic, 0.0)
    marginal_est = float(contrast @ beta)
    marginal_se = float(np.sqrt(contrast @ cov @ contrast))
    z95 = stats.norm.ppf(0.975)

    idx = [names.index(n) for n in interaction_names]
    gamma = beta[idx]
    wald_stat = float(gamma @ np.linalg.solve(cov[np.ix_(idx, idx)], gamma))
    wald = (wald_stat, len(idx), float(stats.chi2.sf(wald_stat, len(idx))))

    out["mixedlm"] = {
        "converged": ml.converged,
        "chair_variance": float(ml.cov_re.iloc[0, 0]),
        "residual_variance": float(ml.scale),
        "party_main_effect": float(ml.params["opp"]),
        "party_main_effect_ci": tuple(ml.conf_int().loc["opp"]),
        "party_main_effect_p": float(ml.pvalues["opp"]),
        "marginal_contrast": marginal_est,
        "marginal_contrast_ci": (marginal_est - z95 * marginal_se, marginal_est + z95 * marginal_se),
        "marginal_contrast_p": float(2 * stats.norm.sf(abs(marginal_est / marginal_se))),
        "joint_wald": wald,
        "n_cells": len(panel),
        "reference_topic": sorted(panel["topic"].unique())[0],
    }

    groups = pd.factorize(panel["sitting"])[0]
    fe = smf.ols("intervention_rate ~ C(topic) * opp + C(chair)", data=panel).fit(
        cov_type="cluster", cov_kwds={"groups": groups}
    )
    fe_names = list(fe.params.index)
    fe_contrast = np.zeros(len(fe_names))
    fe_contrast[fe_names.index("opp")] = 1.0
    for n in [n for n in fe_names if ":opp" in n]:
        topic = re.search(r"C\(topic\)\[T\.(.*)\]:opp", n).group(1)
        fe_contrast[fe_names.index(n)] = weights.get(topic, 0.0)
    fe_marginal = fe.t_test(fe_contrast)
    out["chair_fe"] = {
        "marginal_contrast": float(np.asarray(fe_marginal.effect).ravel()[0]),
        "marginal_contrast_ci": tuple(np.asarray(fe_marginal.conf_int()).ravel()),
        "marginal_contrast_p": float(np.asarray(fe_marginal.pvalue).ravel()[0]),
        "joint_wald": joint_interaction_test(fe),
    }
    return out


def interaction_lr(data: pd.DataFrame) -> float:
    full = smf.glm("y ~ C(topic) * opp + C(chair) + log_words", data=data, family=sm.families.Binomial()).fit()
    restricted = smf.glm("y ~ C(topic) + opp + C(chair) + log_words", data=data, family=sm.families.Binomial()).fit()
    return 2 * (full.llf - restricted.llf)


def permutation_lr_p(data: pd.DataFrame, observed: float, n: int = N_PERMUTATIONS, seed: int = PERMUTATION_SEED) -> float:
    rng = np.random.default_rng(seed)
    spells = data[["speaker_id", "opp"]].drop_duplicates().reset_index(drop=True)
    exceed = 0
    for _ in range(n):
        shuffled = spells.assign(opp_perm=rng.permutation(spells["opp"].to_numpy()))
        permuted = data.merge(shuffled, on=["speaker_id", "opp"]).drop(columns="opp").rename(columns={"opp_perm": "opp"})
        exceed += interaction_lr(permuted) >= observed
    return (exceed + 1) / (n + 1)


def topic_contrasts(utts: pd.DataFrame, outcome: str = "y_any_w3") -> tuple[pd.DataFrame, tuple]:
    counts = utts.groupby(["topic", "opp"]).agg(n=(outcome, "size"), events=(outcome, "sum")).unstack("opp")
    eligible_topics = counts[
        (counts[("n", 0)] >= MIN_TOPIC_UTTS_PER_PARTY)
        & (counts[("n", 1)] >= MIN_TOPIC_UTTS_PER_PARTY)
        & (counts[("events", 0)] >= MIN_TOPIC_EVENTS_PER_PARTY)
        & (counts[("events", 1)] >= MIN_TOPIC_EVENTS_PER_PARTY)
    ].index
    data = utts[utts["topic"].isin(eligible_topics)].copy()
    data["y"] = data[outcome].astype(int)

    result, data = fit_logit(data, "y ~ C(topic) * opp + C(chair) + log_words")
    names = list(result.params.index)

    # Model-based likelihood-ratio test of the interactions, as a check on the
    # cluster-robust Wald test (unstable with many restrictions).
    lr_stat = interaction_lr(data)
    lr_df = int(sum(":opp" in n for n in names))
    print(f"  Running {N_PERMUTATIONS} speaker-level permutations of the interaction LR test...")
    lr = (float(lr_stat), lr_df, float(stats.chi2.sf(lr_stat, lr_df)), permutation_lr_p(data, lr_stat))
    # The two-way clustered covariance is not positive semi-definite over 47
    # restrictions, so the robust Wald test uses sitting-level clustering.
    sitting_result, _ = fit_logit(data, "y ~ C(topic) * opp + C(chair) + log_words", cluster="sitting")
    reference = sorted(eligible_topics)[0]
    z95 = stats.norm.ppf(0.975)

    rows = []
    for topic in sorted(eligible_topics):
        contrast = np.zeros(len(names))
        contrast[names.index("opp")] = 1.0
        if topic != reference:
            contrast[names.index(f"C(topic)[T.{topic}]:opp")] = 1.0
        t = result.t_test(contrast)
        b = float(np.asarray(t.effect).ravel()[0])
        se = float(np.asarray(t.sd).ravel()[0])
        rows.append({
            "topic": topic,
            "n_gov": int(counts.loc[topic, ("n", 0)]),
            "n_opp": int(counts.loc[topic, ("n", 1)]),
            "rate_gov": counts.loc[topic, ("events", 0)] / counts.loc[topic, ("n", 0)],
            "rate_opp": counts.loc[topic, ("events", 1)] / counts.loc[topic, ("n", 1)],
            "OR": np.exp(b),
            "OR_lo95": np.exp(b - z95 * se),
            "OR_hi95": np.exp(b + z95 * se),
            "p": float(np.asarray(t.pvalue).ravel()[0]),
        })
    table = pd.DataFrame(rows)
    table["p_bonferroni"] = multipletests(table["p"], method="bonferroni")[1]
    table["q_bh"] = multipletests(table["p"], method="fdr_bh")[1]
    return table.sort_values("p").reset_index(drop=True), joint_interaction_test(sitting_result), lr


def sensitivity_grid(utts: pd.DataFrame) -> pd.DataFrame:
    base = "y ~ opp + C(topic) + C(chair) + log_words"
    backbench = utts[utts["role_group"] == "backbench"]
    specs = [
        ("Primary: any event, window 3", utts, "y_any_w3", base, TWO_WAY),
        ("Chair rulings only, window 3", utts, "y_ruling_w3", base, TWO_WAY),
        ("Interruptions only, window 3", utts, "y_interruption_w3", base, TWO_WAY),
        ("Any event, next utterance only", utts, "y_any_w1", base, TWO_WAY),
        ("Chair rulings only, next utterance only", utts, "y_ruling_w1", base, TWO_WAY),
        ("Backbench members only", backbench, "y_any_w3", base, TWO_WAY),
        ("Backbench only, chair rulings only", backbench, "y_ruling_w3", base, TWO_WAY),
        ("Excluding RPO members", utts[~utts["is_rpo"]], "y_any_w3", base, TWO_WAY),
        ("Excluding Duval chair", utts[utts["chair"] != "duval"], "y_any_w3", base, TWO_WAY),
        ("Excluding low-confidence topic labels",
         utts[utts["predicted_confidence"] >= LOW_CONFIDENCE_THRESHOLD], "y_any_w3", base, TWO_WAY),
        ("No length (exposure) control", utts, "y_any_w3", "y ~ opp + C(topic) + C(chair)", TWO_WAY),
        ("Sitting fixed effects (within-sitting)", utts, "y_any_w3", "y ~ opp + C(topic) + C(sitting) + log_words", TWO_WAY),
        ("SEs clustered by sitting only", utts, "y_any_w3", base, "sitting"),
        ("SEs clustered by speaker only", utts, "y_any_w3", base, "speaker_id"),
    ]
    for term in ["SIXTH", "SEVENTH", "EIGHTH"]:
        term_utts = utts[utts["term"] == term]
        specs += [
            (f"{term.title()} Assembly: any event", term_utts, "y_any_w3", base, TWO_WAY),
            (f"{term.title()} Assembly: chair rulings only", term_utts, "y_ruling_w3", base, TWO_WAY),
            (f"{term.title()} Assembly: interruptions only", term_utts, "y_interruption_w3", base, TWO_WAY),
        ]

    rows = []
    for label, data, outcome, formula, cluster in specs:
        data = data.assign(y=data[outcome].astype(int))
        if data["chair"].nunique() == 1:
            formula = formula.replace(" + C(chair)", "")
        result, data = fit_logit(data, formula, cluster=cluster)
        # AME for the sitting-FE model is costly and adds little; skip it there.
        rows.append(party_summary(label, result, data, with_ame="C(sitting)" not in formula))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def fmt_or(row) -> str:
    return f"{row['OR']:.3f} [{row['OR_lo95']:.3f}, {row['OR_hi95']:.3f}]"


def fmt_pp(row) -> str:
    if pd.isna(row.get("AME_pp", np.nan)):
        return "-"
    return f"{row['AME_pp']:+.2f} [{row['AME_lo95_pp']:+.2f}, {row['AME_hi95_pp']:+.2f}]"


def write_report(path: Path, utts, panel, original, primary, contrasts, joint, lr, grid) -> None:
    by_party = utts.groupby("opp").agg(
        n=("y_any_w3", "size"), any_w3=("y_any_w3", "mean"),
        ruling_w3=("y_ruling_w3", "mean"), interruption_w3=("y_interruption_w3", "mean"),
    )
    by_chair = (
        utts.groupby("chair")
        .agg(sittings=("sitting", "nunique"), utterances=("sitting", "size"), opposition_share=("opp", "mean"))
        .join(panel.groupby("chair").size().rename("panel_cells"))
    )
    by_role = utts.groupby(["role_group", "opp"]).size().unstack(fill_value=0)
    intervened = utts[utts["y_any_w3"]]
    share_ruling = intervened["y_ruling_w3"].mean()
    share_interruption_only = (intervened["y_interruption_w3"] & ~intervened["y_ruling_w3"]).mean()
    ml, fe = original["mixedlm"], original["chair_fe"]

    lines = [
        "# H1 Reanalysis Report",
        "",
        "Generated by `src/linking_layer/h1_reanalysis.py`. Addresses the external review's H1 points: "
        "panel dimensions, observation unit and dependence, intervention-rate construction, the party main "
        "effect and overall contrast, joint interaction test, chair random vs fixed effects, effect sizes "
        "with intervals, and equivalence testing.",
        "",
        "## A. Observation unit, numerator, denominator",
        "",
        "- **Unit of observation:** a substantive, policy-labelled utterance by a member whose "
        "Government/Opposition status on the sitting date is resolved (ministers, PM and DPM = Government; "
        "Leader of the Opposition = Opposition; backbenchers via the time-aware registry). Chair, "
        "unresolved and surname-collision utterances are excluded.",
        f"- **Analysis-eligible utterances:** {len(utts):,} "
        f"({int(by_party.loc[0, 'n']):,} Government, {int(by_party.loc[1, 'n']):,} Opposition) across "
        f"{utts['sitting'].nunique()} sittings, {utts['topic'].nunique()} policy topics and "
        f"{utts['chair'].nunique()} presiding chairs.",
        f"- **Numerator (original definition):** the utterance is followed, within the next {WINDOW} records of "
        "the same sitting, by a rule-tagged chair ruling **or** a rule-tagged interruption. Interruptions are "
        "Hansard's `(Interruptions)` stage direction and record noise from the floor, not an act of the chair, "
        f"so the chair-rulings-only outcome is reported alongside. Of intervened utterances, {share_ruling:.1%} "
        f"have a chair ruling in the window and {share_interruption_only:.1%} only an interruption.",
        "- **Event attribution:** proximity. The event is attributed to the utterance(s) immediately preceding "
        "it; Hansard does not record the target of a ruling explicitly.",
        "- **Denominator / exposure:** eligible utterances; utterance length (log words) enters the revised "
        "models as an exposure control, since longer contributions have more opportunity to attract an event.",
        f"- **Original 'panel cells':** {len(panel):,} cells = sitting x chair x topic x party, with "
        f"n_utterances >= {MIN_CELL_SIZE}, over {panel['sitting'].nunique()} sittings and "
        f"{panel['topic'].nunique()} topics. The 384-combination bound in the review omits the sitting dimension. "
        f"The cells cover {int(panel.loc[panel.opp == 0, 'n_utterances'].sum()):,} Government and "
        f"{int(panel.loc[panel.opp == 1, 'n_utterances'].sum()):,} Opposition utterances.",
        "- **Correction to the draft:** the 39,396 Government figure is the corpus-wide Layer 0 count of "
        "ministerial utterances, not the H1 sample, and 13,032 Opposition is the pre-cell-filter count. "
        "The figures above replace both.",
        "",
        "### Raw intervention rates",
        "",
        "| Bloc | Utterances | Any event (w=3) | Chair ruling (w=3) | Interruption (w=3) |",
        "|---|---|---|---|---|",
    ]
    for opp, label in [(0, "Government"), (1, "Opposition")]:
        r = by_party.loc[opp]
        lines.append(f"| {label} | {int(r.n):,} | {r.any_w3:.2%} | {r.ruling_w3:.2%} | {r.interruption_w3:.2%} |")

    lines += ["", "### Speaker roles by bloc", "", by_role.rename(columns={0: "Government", 1: "Opposition"}).to_markdown(), ""]
    lines += ["### Chair group sizes", "", by_chair.to_markdown(floatfmt=".3f"), ""]

    lines += [
        "## B. Original specification (Eq. 1) re-examined",
        "",
        f"Cell-level linear MixedLM, `intervention_rate ~ C(topic) * opp`, random intercept for chair "
        f"(n = {ml['n_cells']:,} cells; reference topic: *{ml['reference_topic']}*).",
        "",
        f"- Converged: {ml['converged']}. Chair random-intercept variance: {ml['chair_variance']:.5f} "
        f"(residual variance {ml['residual_variance']:.5f}). With four chairs, one of which presides over "
        f"only {int(by_chair['panel_cells'].min())} cells, this variance is weakly identified, so the "
        "fixed-effects version is reported below.",
        f"- Party main effect (Opposition - Government **in the reference topic only**): "
        f"{ml['party_main_effect']:+.4f} [{ml['party_main_effect_ci'][0]:+.4f}, {ml['party_main_effect_ci'][1]:+.4f}], "
        f"p = {ml['party_main_effect_p']:.3f}.",
        f"- **Overall marginal party contrast** (topic-specific differences averaged with cell-share weights): "
        f"{ml['marginal_contrast']:+.4f} [{ml['marginal_contrast_ci'][0]:+.4f}, {ml['marginal_contrast_ci'][1]:+.4f}], "
        f"p = {ml['marginal_contrast_p']:.3f}.",
        f"- **Joint Wald test of all {ml['joint_wald'][1]} topic x party interactions:** "
        f"chi2 = {ml['joint_wald'][0]:.1f}, p = {ml['joint_wald'][2]:.3f}.",
        f"- Chair **fixed** effects (OLS, SEs clustered by sitting): marginal contrast "
        f"{fe['marginal_contrast']:+.4f} [{fe['marginal_contrast_ci'][0]:+.4f}, {fe['marginal_contrast_ci'][1]:+.4f}], "
        f"p = {fe['marginal_contrast_p']:.3f}; joint interaction test chi2 = {fe['joint_wald'][0]:.1f} "
        f"(df = {fe['joint_wald'][1]}), p = {fe['joint_wald'][2]:.3f}.",
        "- Note for Section 6.2: the chair random intercept does not 'accommodate' the Government/Opposition "
        "sample imbalance. Unequal group sizes need no correction in a regression with a party term; they "
        "only affect the precision of each group's estimate, which the reported intervals reflect.",
        "",
        "## C. Primary revised specification",
        "",
        "Utterance-level logit: `intervened ~ opp + C(topic) + C(chair) + log_words`, SEs two-way clustered "
        "by sitting and speaker (utterances repeat within both). Equivalent to a binomial model of intervention "
        "counts over eligible contributions, and estimates the overall Opposition-Government difference "
        "directly. Speaker identity is the normalised Hansard label; a minister appearing under several "
        "label variants forms several clusters, so these SEs may still be slightly too narrow.",
        "",
        f"- **Odds ratio (Opposition vs Government): {fmt_or(primary)}**, p = {primary['p']:.3f}.",
        f"- **Average marginal effect: {fmt_pp(primary)} percentage points** "
        f"(Government baseline {by_party.loc[0, 'any_w3']:.1%}).",
        f"- Minimum detectable effect (80% power, two-sided alpha = 0.05): {primary['MDE80_pp']:.2f} pp.",
        "- **Equivalence (TOST)**; p < 0.05 means a difference as large as the bound can be rejected:",
        f"  - Benchmark bound +/-{100 * SESOI_PP_BENCHMARK:.0f} pp (Hjorth's ~5 pp copartisan effect): "
        f"p = {primary['tost_p_pp5']:.4f}.",
        f"  - Strict bound +/-{100 * SESOI_PP_STRICT:.0f} pp (~10% of baseline): p = {primary['tost_p_pp2']:.4f}.",
        f"  - Odds-scale bound [{1 / SESOI_OR:.2f}, {SESOI_OR:.2f}]: p = {primary['tost_p_OR']:.4f}.",
        "",
        "## D. Topic-specific party contrasts",
        "",
        f"Model: `intervened ~ C(topic) * opp + C(chair) + log_words`, two-way clustered, restricted to "
        f"{len(contrasts)} topics with >= {MIN_TOPIC_UTTS_PER_PARTY} utterances and >= "
        f"{MIN_TOPIC_EVENTS_PER_PARTY} events in each bloc. Joint tests of the {joint[1]} interactions: "
        f"likelihood-ratio chi2 = {lr[0]:.1f} (df = {lr[1]}), asymptotic p = {lr[2]:.4f}, "
        f"**speaker-level permutation p = {lr[3]:.3f}** ({N_PERMUTATIONS} permutations of party labels across "
        "member party-spells; the reference test, since the asymptotic LR and the sitting-clustered Wald "
        f"ignore within-member dependence); Wald with SEs clustered by sitting chi2 = {joint[0]:.1f}, "
        f"p = {joint[2]:.2g}. The two-way clustered covariance is not positive semi-definite over "
        f"{joint[1]} restrictions, so no two-way joint Wald test is reported. "
        f"Topics with raw p < 0.05: {int((contrasts['p'] < 0.05).sum())}; Bonferroni-significant: "
        f"{int((contrasts['p_bonferroni'] < 0.05).sum())}; BH q < 0.05: {int((contrasts['q_bh'] < 0.05).sum())}. "
        "Full table: `h1_topic_contrasts.csv`.",
        "",
        "| Topic | Gov rate | Opp rate | OR [95% CI] | p | Bonferroni p | BH q |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in contrasts.iterrows():
        lines.append(
            f"| {r.topic} | {r.rate_gov:.1%} | {r.rate_opp:.1%} | {fmt_or(r)} | {r.p:.3f} | "
            f"{r.p_bonferroni:.3f} | {r.q_bh:.3f} |"
        )

    lines += [
        "",
        "## E. Sensitivity analyses (overall party effect)",
        "",
        "All logit, topic and chair fixed effects plus log length unless stated; SEs two-way clustered by "
        "sitting and speaker unless stated. TOST columns test equivalence against the bounds in Section C.",
        "",
        "| Specification | N | N Opp | OR [95% CI] | p | AME pp [95% CI] | MDE80 pp | TOST p (+/-5pp) | TOST p (+/-2pp) | TOST p (OR) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in grid.iterrows():
        def opt(col: str, spec: str) -> str:
            return "-" if pd.isna(r.get(col, np.nan)) else format(r[col], spec)

        lines.append(
            f"| {r.specification} | {r.n_utterances:,} | {r.n_opposition:,} | {fmt_or(r)} | {r.p:.3f} | "
            f"{fmt_pp(r)} | {opt('MDE80_pp', '.2f')} | {opt('tost_p_pp5', '.4f')} | {opt('tost_p_pp2', '.4f')} | "
            f"{r.tost_p_OR:.4f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run(processed_dir: Path) -> None:
    print("Building utterance-level H1 frame...")
    utts = build_utterance_frame(processed_dir)
    print(f"  {len(utts):,} eligible utterances")

    panel = build_cell_panel(utts)
    print(f"  Reconstructed original panel: {len(panel):,} cells")

    print("Replicating original MixedLM and chair-FE version...")
    original = replicate_original(panel)

    print("Fitting sensitivity grid (includes primary specification)...")
    grid = sensitivity_grid(utts)
    primary = grid.iloc[0]

    print("Estimating topic-specific contrasts...")
    contrasts, joint, lr = topic_contrasts(utts)

    write_report(processed_dir / "h1_reanalysis_report.md", utts, panel, original, primary, contrasts, joint, lr, grid)
    contrasts.to_csv(processed_dir / "h1_topic_contrasts.csv", index=False)
    grid.to_csv(processed_dir / "h1_sensitivity.csv", index=False)
    print(f"Wrote {processed_dir / 'h1_reanalysis_report.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    args = parser.parse_args()
    run(Path(args.processed_dir))
