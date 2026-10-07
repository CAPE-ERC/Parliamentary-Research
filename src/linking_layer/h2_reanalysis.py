"""H2 reanalysis (differential deflection), addressing the external review.

Builds on the individual-PQ panel from pq_deflection.py and adds what the
original test lacked:

  1. The question's original addressee (parsed from "asked the <Minister>
     whether ...") and, for transferred questions, the minister it was
     transferred to (parsed clause by clause in pq_deflection.py).
  2. Whether the question sits in the oral or the written-answers section.
  3. An operationalisation of "politically sensitive" that is independent of
     the outcome: two pre-specified accountability lexicons applied to the
     question text, plus a pre-specified topic set. The lexicons and topic
     set below were fixed before transfer rates were examined by sensitivity.
  4. Logit models of transfer with term, section and addressee fixed
     effects, SEs two-way clustered by asker and sitting, reporting ORs,
     average marginal effects, CIs, equivalence tests and group counts.
  5. A transfer typology (acting-minister cover, PM to line minister, line
     minister to PM, lateral) to separate administrative reassignment from
     possible deflection, tested by asker party.

Usage:
    python -m linking_layer.h2_reanalysis
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import fisher_exact

from linking_layer.h1_reanalysis import average_marginal_effect, fit_logit, tost_p

TWO_WAY = ("asker_id", "sitting")
SESOI_PP = 0.01  # about a third of the ~3% baseline transfer rate
MIN_ADDRESSEE_PQS = 30  # rarer addressee titles are pooled into "other"

# Pre-specified sensitivity definitions (fixed before examining outcomes).
CORE_SENSITIVE_TERMS = [
    r"corrupt\w*", r"\bICAC\b", r"\bFCC\b", r"Financial Crimes Commission", r"fraud\w*", r"alleg\w+",
    r"irregularit\w*", r"misuse", r"embezzl\w*", r"money laundering", r"conflict of interest", r"nepotism",
    r"brib\w+", r"kickback", r"scandal", r"malpractice", r"provisional charge", r"misappropriat\w*",
]
EXTENDED_SENSITIVE_TERMS = CORE_SENSITIVE_TERMS + [
    r"procurement", r"tender\w*", r"contracts? (?:was |were )?awarded", r"Director of Audit", r"Audit Report",
    r"Public Accounts Committee", r"inquir\w+", r"enquir\w+", r"investigat\w+", r"arrest\w*",
    r"board members?", r"appoint\w+ of (?:the )?(?:Chairperson|Chairman|Director|CEO|Chief Executive)",
]
SENSITIVE_TOPICS = {
    "Police investigations and criminal cases",
    "Courts, prosecutions and legal cases",
    "Banking, financial sector and public funds",
    "Public finance, banking and funds",
    "MBC broadcasting, news and media governance",
    "CCTV, surveillance and security technology",
    "State land, leases and land administration",
}

ASKED_RE = re.compile(r"asked the (.*?)(?:,?\s+whether\b|,?\s+if\b)", re.IGNORECASE | re.DOTALL)
TITLE_PREFIX_RE = re.compile(r"^(?:the\s+)?(?:Rt\.?\s*)?(?:hon\.?\s*)?(?:Dr\.?\s*)?", re.IGNORECASE)
WRITTEN_MARKER = "WRITTEN ANSWERS TO QUESTIONS"


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def clean_title(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return TITLE_PREFIX_RE.sub("", text).strip()


def office_type(title: str | None) -> str:
    if not isinstance(title, str) or not title:
        return "unknown"
    t = title.lower()
    if re.search(r"\bag\.|\bacting\b", t):
        return "acting"
    if re.search(r"(deputy|vice)[- ]prime minister", t):
        return "deputy_pm"
    if "prime minister" in t:
        return "pm"
    if "minister mentor" in t:
        return "minister_mentor"
    if "attorney general" in t or "minister" in t:
        return "minister"
    return "unknown"


def addressee_key(title: str | None) -> str | None:
    """Normalised first office named, e.g. 'minister of health and wellness'."""
    if not isinstance(title, str) or not title:
        return None
    return clean_title(title).split(",")[0].strip().lower()


def question_text(panel: pd.DataFrame, utterances: pd.DataFrame) -> pd.Series:
    text_lookup = utterances.set_index(["debate_id", "seq_index"])["text"]

    def get(row) -> str:
        text = str(text_lookup.get((row.debate_id, row.header_seq_index), ""))
        idx = text.find(f"(No. B/{row.pq_num})")
        if idx >= 0:
            text = text[idx:]
        if "asked" not in text[:400]:
            text = text + " " + str(text_lookup.get((row.debate_id, row.header_seq_index + 1), ""))
        return text[:2000]

    return panel.apply(get, axis=1)


def matches_any(texts: pd.Series, patterns: list[str]) -> pd.Series:
    return texts.str.contains("|".join(f"(?:{p})" for p in patterns), case=False, regex=True, na=False)


def build_h2_frame(processed_dir: Path) -> pd.DataFrame:
    panel = pd.read_parquet(processed_dir / "pq_deflection_panel.parquet")
    utterances = pd.read_parquet(processed_dir / "utterances.parquet", columns=["debate_id", "seq_index", "text"])

    panel["question_text"] = question_text(panel, utterances)
    panel["addressee_raw"] = panel["question_text"].map(
        lambda t: clean_title(ASKED_RE.search(t).group(1)) if ASKED_RE.search(t) else None
    )
    # Body of the question only (after the addressee), so ministry names in
    # the addressee do not trigger the lexicons.
    panel["question_body"] = panel["question_text"].map(
        lambda t: t[ASKED_RE.search(t).end():] if ASKED_RE.search(t) else t
    )
    panel["addressee_office"] = panel["addressee_raw"].map(office_type)
    panel["addressee_key"] = panel["addressee_raw"].map(addressee_key)

    written_start = (
        utterances[utterances["text"].str.contains(WRITTEN_MARKER, na=False, regex=False)]
        .groupby("debate_id")["seq_index"].min()
    )
    panel["written"] = [
        d in written_start.index and s >= written_start[d]
        for d, s in zip(panel["debate_id"], panel["header_seq_index"])
    ]

    panel["destination_office"] = np.where(
        panel["transferred"], panel["destination"].map(office_type), None
    )

    panel["sensitive_core"] = matches_any(panel["question_body"], CORE_SENSITIVE_TERMS)
    panel["sensitive_extended"] = matches_any(panel["question_body"], EXTENDED_SENSITIVE_TERMS)
    panel["sensitive_topic"] = panel["topic"].isin(SENSITIVE_TOPICS)
    panel["addressed_to_pm"] = panel["addressee_office"].eq("pm")

    resolved = panel[panel["resolved_gov_opp"].notna()].copy()
    resolved["opp"] = (resolved["resolved_gov_opp"] == "opposition").astype(int)
    resolved["y"] = resolved["transferred"].astype(int)
    resolved["asker_id"] = resolved["asker_raw"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    resolved["sitting"] = resolved["debate_id"]
    resolved["term"] = resolved["assembly"]
    resolved["section"] = np.where(resolved["written"], "written", "oral")
    counts = resolved["addressee_key"].value_counts()
    resolved["addressee_fe"] = resolved["addressee_key"].where(
        resolved["addressee_key"].map(counts).fillna(0) >= MIN_ADDRESSEE_PQS, "other"
    ).fillna("other")
    resolved["transfer_type"] = resolved.apply(transfer_type, axis=1)
    resolved["y_substantive"] = (resolved["transferred"] & (resolved["transfer_type"] != "acting_minister_cover")).astype(int)
    return resolved.reset_index(drop=True)


def transfer_type(row) -> str | None:
    if not row["transferred"]:
        return None
    dest = row["destination_office"]
    origin = row["addressee_office"]
    if dest == "acting":
        return "acting_minister_cover"
    if dest in (None, "unknown"):
        return "destination_unparsed"
    if origin == "pm" and dest != "pm":
        return "pm_to_minister"
    if origin != "pm" and dest == "pm":
        return "minister_to_pm"
    return "minister_to_minister"


# --------------------------------------------------------------------------
# Estimation
# --------------------------------------------------------------------------

def summarise_party(label: str, result, data: pd.DataFrame) -> dict:
    z95 = stats.norm.ppf(0.975)
    b, se = result.params["opp"], result.bse["opp"]
    ame, ame_se = average_marginal_effect(result, data)
    return {
        "specification": label,
        "n": int(result.nobs),
        "n_opp": int(data["opp"].sum()),
        "n_transferred": int(data["y"].sum()),
        "n_askers": data["asker_id"].nunique(),
        "n_sittings": data["sitting"].nunique(),
        "OR": np.exp(b), "OR_lo95": np.exp(b - z95 * se), "OR_hi95": np.exp(b + z95 * se),
        "p": result.pvalues["opp"],
        "AME_pp": 100 * ame, "AME_lo95_pp": 100 * (ame - z95 * ame_se), "AME_hi95_pp": 100 * (ame + z95 * ame_se),
        "MDE80_pp": 100 * (z95 + stats.norm.ppf(0.80)) * ame_se,
        "tost_p": tost_p(ame, ame_se, -SESOI_PP, SESOI_PP),
    }


def at_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Questions that could be transferred: withdrawn questions are never
    answered, so they leave the denominator."""
    return df[~df["withdrawn"]]


def top_askers(df: pd.DataFrame, n: int = 5) -> set:
    return set(df[df["opp"] == 1]["asker_id"].value_counts().head(n).index)


def party_models(df: pd.DataFrame) -> pd.DataFrame:
    controls = "C(term) + C(section) + C(addressee_fe)"
    risk = at_risk(df)
    specs = [
        ("Unadjusted, all PQs (original Eq. 2 population)", df, "y ~ opp"),
        ("Unadjusted, withdrawn PQs excluded", risk, "y ~ opp"),
        ("Primary: adjusted (term + section + addressee FE), withdrawn excluded", risk, f"y ~ opp + {controls}"),
        ("Adjusted, all PQs incl. withdrawn", df, f"y ~ opp + {controls}"),
        ("Adjusted, substantive transfers only (acting-minister cover excluded)",
         risk.assign(y=risk["y_substantive"]), f"y ~ opp + {controls}"),
        ("Adjusted, excluding the 5 most frequent Opposition askers",
         risk[~risk["asker_id"].isin(top_askers(risk))], f"y ~ opp + {controls}"),
        ("Adjusted, oral questions only", risk[risk["section"] == "oral"], "y ~ opp + C(term) + C(addressee_fe)"),
        ("Adjusted, written questions only", risk[risk["section"] == "written"], "y ~ opp + C(term) + C(addressee_fe)"),
    ]
    for term in ["SIXTH", "SEVENTH", "EIGHTH"]:
        specs.append((f"Adjusted, {term.title()} Assembly only", risk[risk["term"] == term],
                      "y ~ opp + C(section) + C(addressee_fe)"))
    rows = []
    for label, data, formula in specs:
        result, data = fit_logit(data, formula, cluster=TWO_WAY)
        rows.append(summarise_party(label, result, data))
    return pd.DataFrame(rows)


def sensitivity_models(df: pd.DataFrame) -> pd.DataFrame:
    """For each sensitivity definition: party OR within sensitive and
    non-sensitive questions, and the party x sensitive interaction."""
    z95 = stats.norm.ppf(0.975)
    rows = []
    for flag, label in [
        ("sensitive_core", "Core accountability lexicon"),
        ("sensitive_extended", "Extended accountability lexicon"),
        ("sensitive_topic", "Pre-specified sensitive topics"),
        ("addressed_to_pm", "Addressed to the Prime Minister"),
    ]:
        data = at_risk(df).assign(s=at_risk(df)[flag].astype(int))
        result, data = fit_logit(
            # The PM flag is a function of the addressee, so addressee FE are
            # replaced by the PM flag itself for that definition.
            data,
            "y ~ opp * s + C(term) + C(section)" + ("" if flag == "addressed_to_pm" else " + C(addressee_fe)"),
            cluster=TWO_WAY,
        )
        names = list(result.params.index)
        for s_value, s_label in [(1, "sensitive"), (0, "not sensitive")]:
            contrast = np.zeros(len(names))
            contrast[names.index("opp")] = 1.0
            if s_value:
                contrast[names.index("opp:s")] = 1.0
            t = result.t_test(contrast)
            b = float(np.asarray(t.effect).ravel()[0])
            se = float(np.asarray(t.sd).ravel()[0])
            subset = data[data["s"] == s_value]
            rates = subset.groupby("opp")["y"].agg(["mean", "size", "sum"])
            rows.append({
                "definition": label,
                "subset": s_label,
                "n_gov": int(rates.loc[0, "size"]), "transferred_gov": int(rates.loc[0, "sum"]),
                "rate_gov": rates.loc[0, "mean"],
                "n_opp": int(rates.loc[1, "size"]), "transferred_opp": int(rates.loc[1, "sum"]),
                "rate_opp": rates.loc[1, "mean"],
                "OR": np.exp(b), "OR_lo95": np.exp(b - z95 * se), "OR_hi95": np.exp(b + z95 * se),
                "p": float(np.asarray(t.pvalue).ravel()[0]),
                "interaction_OR": np.exp(result.params["opp:s"]),
                "interaction_p": result.pvalues["opp:s"],
            })
    return pd.DataFrame(rows)


def typology_table(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    transferred = df[df["transferred"]]
    table = pd.crosstab(transferred["transfer_type"], transferred["resolved_gov_opp"])
    _, p, _, _ = stats.chi2_contingency(table)
    return table, float(p)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def fmt_or(r) -> str:
    return f"{r['OR']:.3f} [{r['OR_lo95']:.3f}, {r['OR_hi95']:.3f}]"


def write_report(path: Path, df: pd.DataFrame, party: pd.DataFrame, sens: pd.DataFrame,
                 typology: pd.DataFrame, typology_p: float) -> None:
    contingency = pd.crosstab(df["opp"], df["y"])
    fisher_or, fisher_p = fisher_exact(contingency)
    n_dest = df.loc[df["transferred"], "destination"].notna().sum()
    sens_counts = {c: int(df[c].sum()) for c in ["sensitive_core", "sensitive_extended", "sensitive_topic", "addressed_to_pm"]}

    lines = [
        "# H2 Reanalysis Report",
        "",
        "Generated by `src/linking_layer/h2_reanalysis.py`. Addresses the external review's H2 points: "
        "operationalising 'politically sensitive', contextual controls (portfolio, term, section), "
        "dependence (repeated askers and sittings), group counts, ORs with CIs, and distinguishing "
        "deflection from legitimate portfolio reassignment.",
        "",
        "## Data",
        "",
        f"- PQs with a resolved asker party: {len(df):,} ({int(df.opp.sum()):,} Opposition, "
        f"{int((1 - df.opp).sum()):,} Government) from {df.asker_id.nunique()} askers across "
        f"{df.sitting.nunique()} sittings; {int(df.y.sum()):,} transferred.",
        f"- Addressee parsed for {df.addressee_raw.notna().mean():.1%} of PQs; "
        f"{df.addressee_fe.nunique()} addressee categories (titles with < {MIN_ADDRESSEE_PQS} PQs pooled).",
        f"- Section: {int((df.section == 'oral').sum()):,} oral, {int((df.section == 'written').sum()):,} written. "
        f"Transfer rate oral {df.loc[df.section == 'oral', 'y'].mean():.2%}, "
        f"written {df.loc[df.section == 'written', 'y'].mean():.2%}.",
        f"- Destination minister parsed for {n_dest:,} of {int(df.y.sum()):,} transferred PQs.",
        f"- Withdrawn PQs (named in a withdrawal clause; not transfers, and withdrawn at the asker's "
        f"initiative): {int(df.withdrawn.sum()):,} - Government {df.loc[df.opp == 0, 'withdrawn'].mean():.2%}, "
        f"Opposition {df.loc[df.opp == 1, 'withdrawn'].mean():.2%}. An earlier parser counted these, and the "
        "header of the next question called in the same announcement, as transfers.",
        f"- Fisher's exact (replication): OR = {fisher_or:.3f}, p = {fisher_p:.3f}.",
        "",
        "## Party effect on transfer",
        "",
        f"Logit; SEs two-way clustered by asker and sitting. Equivalence bound +/-{100 * SESOI_PP:.0f} pp "
        "(about a third of the ~3% baseline). Withdrawn questions cannot be transferred, so the primary "
        "specification excludes them from the denominator.",
        "",
        "| Specification | N | N Opp | Transferred | Askers | Sittings | OR [95% CI] | p | AME pp [95% CI] | MDE80 pp | TOST p |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in party.iterrows():
        lines.append(
            f"| {r.specification} | {r.n:,} | {r.n_opp:,} | {r.n_transferred:,} | {r.n_askers} | {r.n_sittings} | "
            f"{fmt_or(r)} | {r.p:.3f} | {r.AME_pp:+.2f} [{r.AME_lo95_pp:+.2f}, {r.AME_hi95_pp:+.2f}] | "
            f"{r.MDE80_pp:.2f} | {r.tost_p:.4f} |"
        )

    lines += [
        "",
        "## Politically sensitive questions",
        "",
        "Sensitivity is defined from the question text or topic, independently of whether the question "
        "was transferred. Definitions (fixed before estimation):",
        "",
        f"- **Core accountability lexicon** ({sens_counts['sensitive_core']:,} PQs): "
        + ", ".join(f"`{t}`" for t in CORE_SENSITIVE_TERMS) + ".",
        f"- **Extended lexicon** ({sens_counts['sensitive_extended']:,} PQs): core plus "
        + ", ".join(f"`{t}`" for t in EXTENDED_SENSITIVE_TERMS[len(CORE_SENSITIVE_TERMS):]) + ".",
        f"- **Pre-specified sensitive topics** ({sens_counts['sensitive_topic']:,} PQs): "
        + "; ".join(sorted(SENSITIVE_TOPICS)) + ".",
        f"- **Addressed to the Prime Minister** ({sens_counts['addressed_to_pm']:,} PQs): a salience proxy.",
        "",
        "Lexicons are applied to the question body after the addressee. Their precision should be checked "
        "on the human-annotated validation sample.",
        "",
        "Model: `y ~ opp * sensitive + C(term) + C(section) + C(addressee_fe)` (addressee FE omitted for the PM definition, which they determine), two-way clustered. The "
        "H2 contrast is the Opposition-Government OR **within sensitive questions**. Withdrawn PQs excluded.",
        "",
        "| Definition | Subset | Gov n (transferred) | Gov rate | Opp n (transferred) | Opp rate | OR [95% CI] | p | Interaction OR (p) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in sens.iterrows():
        lines.append(
            f"| {r.definition} | {r.subset} | {r.n_gov:,} ({r.transferred_gov}) | {r.rate_gov:.2%} | "
            f"{r.n_opp:,} ({r.transferred_opp}) | {r.rate_opp:.2%} | {fmt_or(r)} | {r.p:.3f} | "
            f"{r.interaction_OR:.3f} ({r.interaction_p:.3f}) |"
        )

    shares = typology.div(typology.sum(axis=0), axis=1)
    lines += [
        "",
        "## Transfer typology: administrative reassignment vs possible deflection",
        "",
        "- **acting_minister_cover**: transferred to an Ag./Acting minister standing in for the addressee "
        "(administrative).",
        "- **pm_to_minister**: addressed to the PM, answered by a line minister (delegation to the "
        "responsible portfolio, ordinarily legitimate).",
        "- **minister_to_pm** / **minister_to_minister**: reassignment away from the addressed line "
        "minister; the categories where deflection, if any, would be most plausible.",
        "",
        "| Transfer type | Government | Opposition | Gov share | Opp share |",
        "|---|---|---|---|---|",
    ]
    for t in typology.index:
        lines.append(
            f"| {t} | {typology.loc[t].get('government', 0)} | {typology.loc[t].get('opposition', 0)} | "
            f"{shares.loc[t].get('government', 0):.1%} | {shares.loc[t].get('opposition', 0):.1%} |"
        )
    lines += [
        "",
        f"Chi-square test of transfer type by asker party: p = {typology_p:.3f}.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(processed_dir: Path) -> None:
    print("Building H2 frame (addressee, section, destination, sensitivity)...")
    df = build_h2_frame(processed_dir)
    print(f"  {len(df):,} resolved PQs, {int(df.y.sum()):,} transferred")
    party = party_models(df)
    sens = sensitivity_models(df)
    typology, typology_p = typology_table(df)
    write_report(processed_dir / "h2_reanalysis_report.md", df, party, sens, typology, typology_p)
    party.to_csv(processed_dir / "h2_party_models.csv", index=False)
    sens.to_csv(processed_dir / "h2_sensitive_contrasts.csv", index=False)
    df.drop(columns=["question_text", "question_body"]).to_parquet(processed_dir / "h2_frame.parquet", index=False)
    print(f"Wrote {processed_dir / 'h2_reanalysis_report.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    args = parser.parse_args()
    run(Path(args.processed_dir))
