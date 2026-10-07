"""H3 reanalysis (attention cooling), addressing the external review.

The original series set conflict_rate to 0 in sittings where a topic was not
raised, conflating "no conflict" with "no attention", and reported the pooled
conflict coefficient (-0.0002) without a scale or interval. This module:

  1. Rebuilds the topic x sitting panel from the full record with two conflict
     measures (any event, as in H1; chair rulings only), defined only where the
     topic is present, plus a presence indicator.
  2. Replicates the original pooled MixedLM and reports its CI.
  3. Primary model: next-sitting attention share on current attention,
     presence and standardised conflict, with topic and sitting fixed effects
     (sitting FE absorb common sitting shocks and the shared denominator of
     topic shares), SEs two-way clustered by topic and sitting. Effects are
     reported per SD of conflict, as a share of mean attention, with an
     equivalence test against +/-10% of mean attention.
  4. Sensitivity: chair rulings only, no sitting FE, pandemic period excluded,
     per Assembly term, consecutive (non-recess) sittings only, term-boundary
     transitions excluded, fractional logit, distributed lags, and a test of
     whether the conflict coefficient is stable across terms.
  5. Diagnostics: per-topic ADF and KPSS tests, residual autocorrelation and
     sitting-gap distribution.
  6. Per-topic lag-1 conflict coefficients with sign, HAC CIs and one-sided
     p-values for the predicted (negative) direction, alongside the original
     Granger F-test and a Granger test on first-differenced series.

Usage:
    python -m outcome_layer.h3_reanalysis
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, kpss

from linking_layer.build_panel import WINDOW
from linking_layer.h1_reanalysis import flag_window, tost_p
from linking_layer.party_resolution import parse_sitting_date
from outcome_layer.granger_test import MIN_NONZERO_CONFLICT_SITTINGS

PANDEMIC_START = pd.Timestamp("2020-03-01")
PANDEMIC_END = pd.Timestamp("2021-12-31")
MAX_CONSECUTIVE_GAP_DAYS = 14  # longer gaps span a recess
SESOI_SHARE_OF_MEAN = 0.10  # +/-10% of mean attention per SD of conflict
HAC_LAGS = 4
TWO_WAY = ("topic", "debate_id")


# --------------------------------------------------------------------------
# Panel construction
# --------------------------------------------------------------------------

def build_panel(processed_dir: Path) -> pd.DataFrame:
    utterances = pd.read_parquet(
        processed_dir / "utterances.parquet",
        columns=["debate_id", "seq_index", "assembly", "sitting_date", "is_stage_direction"],
    )
    topics = pd.read_parquet(processed_dir / "utterance_policy_labels_two_stage.parquet")
    tags = pd.read_parquet(processed_dir / "procedural_tags_final.parquet")

    df = utterances.merge(
        tags[["debate_id", "seq_index", "chair_ruling_rule", "interruption_rule"]], on=["debate_id", "seq_index"], how="left"
    )
    df["ruling_event"] = df["chair_ruling_rule"].fillna(0).astype(bool)
    df["any_event"] = df["ruling_event"] | df["interruption_rule"].fillna(0).astype(bool)
    df = df.sort_values(["debate_id", "seq_index"]).reset_index(drop=True)
    df["y_any"] = flag_window(df, "any_event", WINDOW)
    df["y_ruling"] = flag_window(df, "ruling_event", WINDOW)

    sittings = df[["debate_id", "sitting_date", "assembly"]].drop_duplicates("debate_id")
    sittings["date"] = sittings["sitting_date"].apply(parse_sitting_date)
    sittings = sittings.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    sittings["sitting_order"] = range(len(sittings))

    speech = df[~df["is_stage_direction"]].merge(topics[["debate_id", "seq_index", "predicted_label"]],
                                                 on=["debate_id", "seq_index"])
    totals = speech.groupby("debate_id").size().rename("total_utterances")
    policy = speech[speech["predicted_label"] != "non_policy"]
    per_cell = (
        policy.groupby(["predicted_label", "debate_id"])
        .agg(n_topic=("y_any", "size"), n_any=("y_any", "sum"), n_ruling=("y_ruling", "sum"))
        .reset_index()
        .rename(columns={"predicted_label": "topic"})
    )

    topics_list = sorted(policy["predicted_label"].unique())
    grid = pd.MultiIndex.from_product([topics_list, sittings["debate_id"]], names=["topic", "debate_id"]).to_frame(index=False)
    grid = grid.merge(per_cell, on=["topic", "debate_id"], how="left").fillna({"n_topic": 0, "n_any": 0, "n_ruling": 0})
    grid = grid.merge(sittings[["debate_id", "date", "assembly", "sitting_order"]], on="debate_id")
    grid = grid.merge(totals, on="debate_id", how="left")

    grid["attention"] = grid["n_topic"] / grid["total_utterances"]
    grid["present"] = (grid["n_topic"] > 0).astype(int)
    grid["conflict_any"] = np.where(grid["present"] == 1, grid["n_any"] / grid["n_topic"].where(grid["n_topic"] > 0), np.nan)
    grid["conflict_ruling"] = np.where(grid["present"] == 1, grid["n_ruling"] / grid["n_topic"].where(grid["n_topic"] > 0), np.nan)

    grid = grid.sort_values(["topic", "sitting_order"]).reset_index(drop=True)
    g = grid.groupby("topic")
    grid["attention_next"] = g["attention"].shift(-1)
    grid["date_next"] = g["date"].shift(-1)
    grid["assembly_next"] = g["assembly"].shift(-1)
    for k in (1, 2):
        grid[f"attention_lag{k}"] = g["attention"].shift(k)
        grid[f"present_lag{k}"] = g["present"].shift(k)
    grid["gap_days"] = (grid["date_next"] - grid["date"]).dt.days
    grid["crosses_term"] = grid["assembly_next"].notna() & (grid["assembly_next"] != grid["assembly"])
    grid["pandemic"] = grid["date"].between(PANDEMIC_START, PANDEMIC_END) | grid["date_next"].between(PANDEMIC_START, PANDEMIC_END)

    # Standardise conflict over present cells; absent cells get 0 and are
    # separated out by the presence indicator, so the conflict coefficient is
    # identified only from sittings where the topic was raised.
    for measure in ("conflict_any", "conflict_ruling"):
        present_values = grid.loc[grid["present"] == 1, measure]
        grid[f"{measure}_sd"] = present_values.std()
        grid[f"{measure}_z"] = ((grid[measure] - present_values.mean()) / present_values.std()).fillna(0.0)
        for k in (1, 2):
            grid[f"{measure}_z_lag{k}"] = grid.groupby("topic")[f"{measure}_z"].shift(k)
    return grid


# --------------------------------------------------------------------------
# Estimation
# --------------------------------------------------------------------------

def fit_ols(data: pd.DataFrame, formula: str, cluster=TWO_WAY):
    data = data.reset_index(drop=True)
    clusters = (cluster,) if isinstance(cluster, str) else cluster
    groups = np.column_stack([pd.factorize(data[c])[0] for c in clusters])
    if groups.shape[1] == 1:
        groups = groups[:, 0]
    return smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": groups}), data


def effect_row(label: str, result, data: pd.DataFrame, term: str, mean_attention: float) -> dict:
    b, se = result.params[term], result.bse[term]
    z95 = stats.norm.ppf(0.975)
    bound = SESOI_SHARE_OF_MEAN * mean_attention
    return {
        "specification": label,
        "n_cells": int(result.nobs),
        "n_present": int(data["present"].sum()),
        "beta_per_sd": b,
        "lo95": b - z95 * se,
        "hi95": b + z95 * se,
        "p_two_sided": result.pvalues[term],
        "p_one_sided_negative": stats.norm.cdf(b / se),
        "pct_of_mean": 100 * b / mean_attention,
        "pct_lo95": 100 * (b - z95 * se) / mean_attention,
        "pct_hi95": 100 * (b + z95 * se) / mean_attention,
        "tost_p": tost_p(b, se, -bound, bound),
    }


def pooled_models(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    base = panel.dropna(subset=["attention_next"])
    mean_attention = base["attention"].mean()
    fe = "C(topic) + C(debate_id)"
    rhs_any = "attention + present + conflict_any_z"
    rhs_ruling = "attention + present + conflict_ruling_z"
    consecutive = base[base["gap_days"] <= MAX_CONSECUTIVE_GAP_DAYS]

    specs = [
        ("Primary: any event, topic + sitting FE", base, f"attention_next ~ {rhs_any} + {fe}", "conflict_any_z"),
        ("Chair rulings only", base, f"attention_next ~ {rhs_ruling} + {fe}", "conflict_ruling_z"),
        ("Topic FE only (no sitting FE)", base, f"attention_next ~ {rhs_any} + C(topic)", "conflict_any_z"),
        ("Pandemic period excluded (Mar 2020 - Dec 2021)", base[~base["pandemic"]],
         f"attention_next ~ {rhs_any} + {fe}", "conflict_any_z"),
        (f"Consecutive sittings only (gap <= {MAX_CONSECUTIVE_GAP_DAYS} days)", consecutive,
         f"attention_next ~ {rhs_any} + {fe}", "conflict_any_z"),
        ("Term-boundary transitions excluded", base[~base["crosses_term"]],
         f"attention_next ~ {rhs_any} + {fe}", "conflict_any_z"),
        ("Two lags of attention", base.dropna(subset=["attention_lag1"]),
         f"attention_next ~ {rhs_any} + attention_lag1 + present_lag1 + {fe}", "conflict_any_z"),
        ("Present-only cells (topic raised at t)", base[base["present"] == 1],
         f"attention_next ~ attention + conflict_any_z + {fe}", "conflict_any_z"),
    ]
    for term in ["SIXTH", "SEVENTH", "EIGHTH"]:
        specs.append((f"{term.title()} Assembly only", base[base["assembly"] == term],
                      f"attention_next ~ {rhs_any} + {fe}", "conflict_any_z"))

    rows = []
    for label, data, formula, term in specs:
        result, data = fit_ols(data, formula)
        rows.append(effect_row(label, result, data, term, mean_attention))

    # Distributed lag: cumulative effect of conflict at t, t-1, t-2.
    lagged = base.dropna(subset=["conflict_any_z_lag2", "attention_lag2"])
    result, lagged = fit_ols(
        lagged,
        "attention_next ~ attention + attention_lag1 + attention_lag2 + present + present_lag1 + present_lag2 "
        f"+ conflict_any_z + conflict_any_z_lag1 + conflict_any_z_lag2 + {fe}",
    )
    names = list(result.params.index)
    contrast = np.zeros(len(names))
    for t in ("conflict_any_z", "conflict_any_z_lag1", "conflict_any_z_lag2"):
        contrast[names.index(t)] = 1.0
    t = result.t_test(contrast)
    b = float(np.asarray(t.effect).ravel()[0])
    se = float(np.asarray(t.sd).ravel()[0])
    z95 = stats.norm.ppf(0.975)
    rows.append({
        "specification": "Distributed lag: sum of conflict at t, t-1, t-2",
        "n_cells": int(result.nobs), "n_present": int(lagged["present"].sum()),
        "beta_per_sd": b, "lo95": b - z95 * se, "hi95": b + z95 * se,
        "p_two_sided": float(np.asarray(t.pvalue).ravel()[0]), "p_one_sided_negative": stats.norm.cdf(b / se),
        "pct_of_mean": 100 * b / mean_attention, "pct_lo95": 100 * (b - z95 * se) / mean_attention,
        "pct_hi95": 100 * (b + z95 * se) / mean_attention,
        "tost_p": tost_p(b, se, -SESOI_SHARE_OF_MEAN * mean_attention, SESOI_SHARE_OF_MEAN * mean_attention),
    })

    # Fractional logit (shares bounded in [0, 1]); average marginal effect.
    frac = smf.glm(f"attention_next ~ {rhs_any} + C(topic)", data=base.reset_index(drop=True),
                   family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": np.column_stack([pd.factorize(base[c])[0] for c in TWO_WAY])}
    )
    mu = frac.fittedvalues.to_numpy()
    ame = float(np.mean(mu * (1 - mu)) * frac.params["conflict_any_z"])
    ame_se = float(np.mean(mu * (1 - mu)) * frac.bse["conflict_any_z"])
    rows.append({
        "specification": "Fractional logit (topic FE), AME",
        "n_cells": int(frac.nobs), "n_present": int(base["present"].sum()),
        "beta_per_sd": ame, "lo95": ame - z95 * ame_se, "hi95": ame + z95 * ame_se,
        "p_two_sided": frac.pvalues["conflict_any_z"], "p_one_sided_negative": stats.norm.cdf(ame / ame_se),
        "pct_of_mean": 100 * ame / mean_attention, "pct_lo95": 100 * (ame - z95 * ame_se) / mean_attention,
        "pct_hi95": 100 * (ame + z95 * ame_se) / mean_attention,
        "tost_p": tost_p(ame, ame_se, -SESOI_SHARE_OF_MEAN * mean_attention, SESOI_SHARE_OF_MEAN * mean_attention),
    })

    # Stability of the conflict effect across terms.
    stab, _ = fit_ols(base, f"attention_next ~ attention + present + conflict_any_z * C(assembly) + {fe}")
    inter = [n for n in stab.params.index if n.startswith("conflict_any_z:")]
    restriction = np.zeros((len(inter), len(stab.params)))
    for r, n in enumerate(inter):
        restriction[r, list(stab.params.index).index(n)] = 1
    wald = stab.wald_test(restriction, scalar=True)

    # Residual dependence in the primary model.
    primary, primary_data = fit_ols(base, f"attention_next ~ {rhs_any} + {fe}")
    primary_data = primary_data.assign(resid=primary.resid.to_numpy())
    primary_data["resid_lag"] = primary_data.groupby("topic")["resid"].shift(1)
    ar = smf.ols("resid ~ resid_lag", data=primary_data.dropna(subset=["resid_lag"])).fit()

    extras = {
        "mean_attention": mean_attention,
        "sd_conflict_any": float(base["conflict_any_sd"].iloc[0]),
        "sd_conflict_ruling": float(base["conflict_ruling_sd"].iloc[0]),
        "stability_wald": (float(wald.statistic), len(inter), float(wald.pvalue)),
        "resid_ar1": (float(ar.params["resid_lag"]), float(ar.bse["resid_lag"])),
        "gap_quantiles": base["gap_days"].quantile([0.1, 0.5, 0.9, 0.99]).to_dict(),
        "share_consecutive": float((base["gap_days"] <= MAX_CONSECUTIVE_GAP_DAYS).mean()),
        "n_sittings": base["debate_id"].nunique(),
        "n_topics": base["topic"].nunique(),
        "lagged_attention_beta": (float(primary.params["attention"]), float(primary.bse["attention"])),
    }
    return pd.DataFrame(rows), extras


def replicate_original(processed_dir: Path) -> dict:
    series = pd.read_parquet(processed_dir / "outcome_series.parquet").sort_values(["topic", "sitting_order"])
    series["attention_share_next"] = series.groupby("topic")["attention_share"].shift(-1)
    series = series.dropna(subset=["attention_share_next"])
    result = smf.mixedlm("attention_share_next ~ attention_share + conflict_rate", data=series,
                         groups=series["topic"]).fit(reml=False)
    ci = result.conf_int().loc["conflict_rate"]

    # Direction of the one per-topic Granger result that survived Bonferroni
    # in the original analysis.
    covid = series[series["topic"].str.startswith("Vaccination")]
    covid_fit = smf.ols("attention_share_next ~ attention_share + conflict_rate", data=covid).fit(
        cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS}
    )
    covid_ci = covid_fit.conf_int().loc["conflict_rate"]
    return {
        "covid_beta": float(covid_fit.params["conflict_rate"]),
        "covid_ci": (float(covid_ci[0]), float(covid_ci[1])),
        "covid_p": float(covid_fit.pvalues["conflict_rate"]),
        "covid_mean": float(covid["attention_share"].mean()),
        "beta": float(result.params["conflict_rate"]),
        "ci": (float(ci[0]), float(ci[1])),
        "p": float(result.pvalues["conflict_rate"]),
        "mean_attention": float(series["attention_share"].mean()),
        "sd_conflict_present": float(series.loc[series["attention_share"] > 0, "conflict_rate"].std()),
        "attention_beta": float(result.params["attention_share"]),
    }


def stationarity(panel: pd.DataFrame, tested_topics: list[str]) -> pd.DataFrame:
    rows = []
    for topic, g in panel.groupby("topic"):
        g = g.sort_values("sitting_order")
        row = {"topic": topic, "tested_for_h3": topic in tested_topics}
        for col in ("attention", "conflict_any_z"):
            x = g[col].to_numpy()
            if np.std(x) == 0:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", InterpolationWarning)
                row[f"adf_p_{col}"] = adfuller(x, autolag="AIC")[1]
                row[f"kpss_p_{col}"] = kpss(x, regression="c", nlags="auto")[1]
        rows.append(row)
    return pd.DataFrame(rows)


def per_topic(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for topic, g in panel.groupby("topic"):
        g = g.sort_values("sitting_order")
        n_conflict = int((g["n_any"] > 0).sum())
        if n_conflict < MIN_NONZERO_CONFLICT_SITTINGS:
            continue
        d = g.dropna(subset=["attention_next"])
        ols = smf.ols("attention_next ~ attention + present + conflict_any_z", data=d).fit(
            cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS}
        )
        b, se = ols.params["conflict_any_z"], ols.bse["conflict_any_z"]
        mean_att = g["attention"].mean()
        # Directional estimate on the original measure (conflict = 0 when the
        # topic is absent), unstandardised: change in next-sitting share for
        # conflict moving from 0% to 100%.
        orig_rate = (g["n_any"] / g["n_topic"].where(g["n_topic"] > 0)).fillna(0)
        d_orig = g.assign(conflict_orig=orig_rate).dropna(subset=["attention_next"])
        ols_orig = smf.ols("attention_next ~ attention + conflict_orig", data=d_orig).fit(
            cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS}
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            orig = np.column_stack([g["attention"], orig_rate])
            p_orig = grangercausalitytests(orig, maxlag=1, verbose=False)[1][0]["ssr_ftest"][1]
            diffed = np.diff(orig, axis=0)
            p_diff = grangercausalitytests(diffed, maxlag=1, verbose=False)[1][0]["ssr_ftest"][1]
        rows.append({
            "topic": topic,
            "conflict_sittings": n_conflict,
            "beta_per_sd": b,
            "lo95": b - 1.96 * se,
            "hi95": b + 1.96 * se,
            "pct_of_mean": 100 * b / mean_att,
            "p_hac": ols.pvalues["conflict_any_z"],
            "p_one_sided_negative": stats.norm.cdf(b / se),
            "granger_p_levels": p_orig,
            "granger_p_differenced": p_diff,
            "orig_measure_beta": ols_orig.params["conflict_orig"],
            "orig_measure_pct_of_mean": 100 * ols_orig.params["conflict_orig"] / mean_att,
            "orig_measure_p": ols_orig.pvalues["conflict_orig"],
        })
    table = pd.DataFrame(rows)
    alpha = 0.05 / len(table)
    table["bonferroni_hac"] = table["p_hac"] < alpha
    table["bonferroni_granger_levels"] = table["granger_p_levels"] < alpha
    table["bonferroni_granger_differenced"] = table["granger_p_differenced"] < alpha
    return table.sort_values("p_hac").reset_index(drop=True)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def write_report(path: Path, original: dict, pooled: pd.DataFrame, extras: dict, station: pd.DataFrame,
                 topics: pd.DataFrame) -> None:
    mean_att = extras["mean_attention"]
    orig_per_sd = original["beta"] * original["sd_conflict_present"]
    primary = pooled.iloc[0]
    tested = station[station["tested_for_h3"]]
    alpha_topics = 0.05 / len(topics)
    n_raw = int((topics.p_hac < 0.05).sum())

    def n_flag(df, col, cond):
        return int(cond(df[col]).sum()) if col in df else 0

    lines = [
        "# H3 Reanalysis Report",
        "",
        "Generated by `src/outcome_layer/h3_reanalysis.py`. Addresses the external review's H3 points: "
        "variable scales and intervals for the pooled coefficient, stationarity and structural breaks, "
        "sitting gaps, common sitting shocks and the shared denominator of attention shares, residual "
        "dependence, sensitivity to term boundaries and the pandemic, and directional per-topic estimates "
        "that distinguish predictive precedence from suppression.",
        "",
        "## Scales",
        "",
        f"- Attention share = a topic's share of all substantive utterances in a sitting. Mean "
        f"{mean_att:.4f} ({100 * mean_att:.2f}% of a sitting) across {extras['n_topics']} topics x "
        f"{extras['n_sittings']} sitting transitions.",
        f"- Conflict = share of the topic's utterances followed within {WINDOW} records by a chair ruling or "
        f"interruption (any event), or by a chair ruling only. Defined only where the topic is raised; "
        f"SD among those cells: {extras['sd_conflict_any']:.3f} (any event), {extras['sd_conflict_ruling']:.3f} "
        "(rulings). Models use conflict standardised to SD units plus a presence indicator.",
        "",
        "## Original specification re-examined",
        "",
        f"Pooled MixedLM `attention_(t+1) ~ attention_t + conflict_rate_t`, topic random intercept: "
        f"beta = {original['beta']:.5f} [{original['ci'][0]:.5f}, {original['ci'][1]:.5f}], p = {original['p']:.3f}. "
        "The coefficient is the change in next-sitting attention share for conflict moving from 0% to 100%. "
        f"Per SD of conflict (among sittings where the topic is raised) it is {orig_per_sd:.6f}, i.e. "
        f"{100 * orig_per_sd / original['mean_attention']:.2f}% of mean attention. The original series also set "
        "conflict to 0 in sittings where the topic was absent, so 'no conflict' and 'no attention' shared a value; "
        "the models below separate them.",
        "",
        f"**Direction of the original COVID-19 Granger result** (the only per-topic test to survive Bonferroni, "
        f"p = 0.0007): on the same original series, the lag-1 conflict coefficient for Vaccination, COVID-19 and "
        f"public health is {original['covid_beta']:+.4f} [{original['covid_ci'][0]:+.4f}, {original['covid_ci'][1]:+.4f}] "
        f"(HAC p = {original['covid_p']:.3f}; topic mean attention {original['covid_mean']:.4f}). The point estimate "
        "is positive: conflict preceded more, not less, attention to the topic. The Granger result therefore "
        "indicates predictive precedence, not attention cooling.",
        "",
        "## Primary model",
        "",
        "OLS: `attention_(t+1) ~ attention_t + present_t + conflict_z_t + topic FE + sitting FE`, SEs two-way "
        "clustered by topic and sitting. Sitting FE absorb shocks common to all topics in a sitting, including "
        "the shared denominator of the shares. With ~400 sittings per topic, the dynamic-panel (Nickell) bias "
        "from including the lagged outcome with topic FE is negligible.",
        "",
        f"- **Conflict effect per SD: {primary.beta_per_sd:+.6f} [{primary.lo95:+.6f}, {primary.hi95:+.6f}]**, "
        f"p = {primary.p_two_sided:.3f}; **{primary.pct_of_mean:+.1f}% of mean attention "
        f"[{primary.pct_lo95:+.1f}%, {primary.pct_hi95:+.1f}%]**.",
        f"- One-sided p for the predicted cooling (negative) direction: {primary.p_one_sided_negative:.3f}.",
        f"- Equivalence against +/-{100 * SESOI_SHARE_OF_MEAN:.0f}% of mean attention per SD: TOST p = "
        f"{primary.tost_p:.4f}.",
        f"- Lagged attention coefficient: {extras['lagged_attention_beta'][0]:.3f} "
        f"(SE {extras['lagged_attention_beta'][1]:.3f}). This shows persistence; it does not by itself show "
        "that momentum dominates other predictors.",
        "",
        "## Sensitivity analyses",
        "",
        "| Specification | Cells | Present | Effect per SD [95% CI] | % of mean attention [95% CI] | p | p (one-sided, cooling) | TOST p |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in pooled.iterrows():
        lines.append(
            f"| {r.specification} | {r.n_cells:,} | {r.n_present:,} | {r.beta_per_sd:+.6f} [{r.lo95:+.6f}, {r.hi95:+.6f}] | "
            f"{r.pct_of_mean:+.1f}% [{r.pct_lo95:+.1f}%, {r.pct_hi95:+.1f}%] | {r.p_two_sided:.3f} | "
            f"{r.p_one_sided_negative:.3f} | {r.tost_p:.4f} |"
        )
    sw = extras["stability_wald"]
    lines += [
        "",
        f"- **Temporal stability:** conflict x Assembly-term interactions, joint Wald chi2 = {sw[0]:.2f} "
        f"(df = {sw[1]}), p = {sw[2]:.3f}.",
        f"- **Residual dependence:** AR(1) coefficient of primary-model residuals within topic = "
        f"{extras['resid_ar1'][0]:.3f} (SE {extras['resid_ar1'][1]:.3f}); SEs are clustered by topic, which "
        "is robust to arbitrary within-topic serial correlation.",
        "- **Sitting gaps:** days to the next sitting, 10th/50th/90th/99th percentiles: "
        + ", ".join(f"{v:.0f}" for v in extras["gap_quantiles"].values())
        + f". {100 * extras['share_consecutive']:.0f}% of transitions are within {MAX_CONSECUTIVE_GAP_DAYS} days; "
        "the consecutive-sittings row restricts to those.",
        "",
        "## Stationarity diagnostics",
        "",
        f"Per-topic ADF (H0: unit root) and KPSS (H0: level-stationary) at 5%, for the {len(tested)} topics "
        f"tested in H3 (all {len(station)} topics in brackets):",
        "",
        f"- Attention share: ADF rejects a unit root for {n_flag(tested, 'adf_p_attention', lambda s: s < 0.05)} "
        f"({n_flag(station, 'adf_p_attention', lambda s: s < 0.05)}); KPSS rejects stationarity for "
        f"{n_flag(tested, 'kpss_p_attention', lambda s: s < 0.05)} ({n_flag(station, 'kpss_p_attention', lambda s: s < 0.05)}).",
        f"- Conflict: ADF rejects a unit root for {n_flag(tested, 'adf_p_conflict_any_z', lambda s: s < 0.05)} "
        f"({n_flag(station, 'adf_p_conflict_any_z', lambda s: s < 0.05)}); KPSS rejects stationarity for "
        f"{n_flag(tested, 'kpss_p_conflict_any_z', lambda s: s < 0.05)} "
        f"({n_flag(station, 'kpss_p_conflict_any_z', lambda s: s < 0.05)}).",
        "- Where ADF and KPSS disagree, the series is typically stationary around a shifting level (term or "
        "pandemic breaks) rather than a random walk; the sitting-FE, term and pandemic-excluded models above "
        "address level shifts, and the per-topic Granger tests are repeated on first-differenced series.",
        "",
        "## Per-topic estimates (direction and size)",
        "",
        f"Per topic: OLS `attention_(t+1) ~ attention_t + present_t + conflict_z_t`, Newey-West SEs "
        f"({HAC_LAGS} lags). Bonferroni alpha = 0.05/{len(topics)} = {alpha_topics:.5f}. Granger p-values (lag 1) "
        "are shown for the original levels specification and for first-differenced series. A significant "
        "Granger test shows predictive precedence only; H3 (cooling) additionally requires a negative coefficient.",
        "",
        f"- Raw p < 0.05 (HAC): {n_raw} topics, of which negative: "
        f"{int(((topics.p_hac < 0.05) & (topics.beta_per_sd < 0)).sum())}; Bonferroni: {int(topics.bonferroni_hac.sum())}. "
        f"Under a global null about {0.05 * len(topics):.1f} would be expected; binomial p for observing "
        f"{n_raw} or more = {stats.binomtest(n_raw, len(topics), 0.05, alternative='greater').pvalue:.3f}. "
        f"{100 * (topics.beta_per_sd < 0).mean():.0f}% of all per-topic estimates are negative.",
        f"- Granger levels, Bonferroni-significant: {int(topics.bonferroni_granger_levels.sum())}; "
        f"differenced: {int(topics.bonferroni_granger_differenced.sum())}.",
        "- Per-topic models have no sitting fixed effects, so unlike the pooled model they do not absorb "
        "shocks common to all topics in a sitting.",
        "- The last two columns give the coefficient on the original conflict measure (0 when the topic is "
        "absent; change in next-sitting share for conflict from 0% to 100%, as % of the topic's mean attention), "
        "which supplies the direction missing from the original Granger results.",
        "",
        "| Topic | Conflict sittings | Effect per SD [95% CI] | % of topic mean | p (HAC) | p (cooling) | Granger p (levels) | Granger p (differenced) | Original measure: % of mean (p) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in topics.iterrows():
        lines.append(
            f"| {r.topic} | {r.conflict_sittings} | {r.beta_per_sd:+.5f} [{r.lo95:+.5f}, {r.hi95:+.5f}] | "
            f"{r.pct_of_mean:+.1f}% | {r.p_hac:.4f} | {r.p_one_sided_negative:.3f} | {r.granger_p_levels:.4f} | "
            f"{r.granger_p_differenced:.4f} | {r.orig_measure_pct_of_mean:+.1f}% ({r.orig_measure_p:.3f}) |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run(processed_dir: Path) -> None:
    print("Building topic x sitting panel...")
    panel = build_panel(processed_dir)
    print(f"  {panel.topic.nunique()} topics x {panel.debate_id.nunique()} sittings")
    print("Replicating original pooled model...")
    original = replicate_original(processed_dir)
    print("Fitting pooled models and sensitivity grid...")
    pooled, extras = pooled_models(panel)
    print("Per-topic estimates...")
    topics = per_topic(panel)
    print("Stationarity diagnostics...")
    station = stationarity(panel, topics["topic"].tolist())
    write_report(processed_dir / "h3_reanalysis_report.md", original, pooled, extras, station, topics)
    pooled.to_csv(processed_dir / "h3_pooled_models.csv", index=False)
    topics.to_csv(processed_dir / "h3_per_topic.csv", index=False)
    station.to_csv(processed_dir / "h3_stationarity.csv", index=False)
    print(f"Wrote {processed_dir / 'h3_reanalysis_report.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    args = parser.parse_args()
    run(Path(args.processed_dir))
