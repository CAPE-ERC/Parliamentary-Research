"""Scores completed annotation workbooks against the model outputs.

Takes one completed copy of annotation_workbook.xlsx per annotator (and
optionally an adjudicated copy). Reports:

  - inter-annotator agreement (percent agreement and Cohen's kappa) on the
    double-coded rows of each sheet;
  - topic classifier accuracy, Policy-gate precision/recall and per-class
    precision/recall, overall and by language, bloc and outlier status;
  - procedural tagger precision/recall per tag, for the rule tagger and for
    the downstream value used in the paper (Table 2);
  - chair attribution of H1 intervention windows, by bloc and event type;
  - precision/recall of the H2 sensitivity lexicons;
  - accuracy of the clause-level PQ transfer/withdrawal parser.

Estimates use the sampling weights in the key so they generalise to the
corpus; per-class precision is within the predicted class.

Usage:
    python -m validation.score_annotations --annotators A.xlsx B.xlsx [--adjudicated C.xlsx]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from topic_layer.train_classifier import NON_POLICY_LABEL
from validation.annotation_sample import TAGS, TASKS

# Downstream values used in the paper (Table 2).
DOWNSTREAM = {"interruption": "rule", "withdrawal_request": "combined", "chair_ruling": "rule",
              "so_citation": "rule", "pnq_transfer": "combined"}


def as_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["true", "1", "1.0", "yes"])


def read_annotations(path: Path) -> dict[str, pd.DataFrame]:
    sheets = pd.read_excel(path, sheet_name=list(TASKS))
    return {name: df.set_index("item_id") for name, df in sheets.items()}


def agreement(a: pd.DataFrame, b: pd.DataFrame, fields: list[str]) -> list[dict]:
    rows = []
    both = a.index.intersection(b.index)
    both = [i for i in both if str(a.at[i, "double_code"]) == "yes"]
    for f in fields:
        x, y = a.loc[both, f], b.loc[both, f]
        ok = x.notna() & y.notna()
        if ok.sum() == 0:
            continue
        x, y = x[ok].astype(str), y[ok].astype(str)
        rows.append({"field": f, "n": int(ok.sum()), "agreement": float((x == y).mean()),
                     "kappa": float(cohen_kappa_score(x, y)) if x.nunique() + y.nunique() > 2 else float("nan")})
    return rows


def gold_labels(annotators: list[dict], adjudicated: dict | None, task: str, fields: list[str]) -> pd.DataFrame:
    """Adjudicated value where given, otherwise the first annotator who coded the item."""
    frames = ([adjudicated] if adjudicated else []) + annotators
    gold = None
    for f in frames:
        sheet = f[task][fields]
        gold = sheet if gold is None else gold.combine_first(sheet)
    return gold


def weighted_pr(truth: pd.Series, pred: pd.Series, weight: pd.Series) -> dict:
    tp = (weight * (truth & pred)).sum()
    fp = (weight * (~truth & pred)).sum()
    fn = (weight * (truth & ~pred)).sum()
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")
    return {"n": int(len(truth)), "n_true": int(truth.sum()), "precision": precision, "recall": recall, "f1": f1}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def score_topic(key: pd.DataFrame, gold: pd.DataFrame) -> list[str]:
    d = key.join(gold, how="inner").dropna(subset=["topic_label"])
    d["correct"] = d["topic_label"] == d["predicted_label"]
    w = d["weight"].astype(float)
    lines = ["## Topic classification", "",
             f"Coded items: {len(d)}. Weighted accuracy: {(w * d.correct).sum() / w.sum():.1%}.", ""]
    gate = weighted_pr(d["topic_label"] != NON_POLICY_LABEL, d["predicted_label"] != NON_POLICY_LABEL, w)
    lines += [f"Policy gate: precision {gate['precision']:.1%}, recall {gate['recall']:.1%}, F1 {gate['f1']:.3f}.", ""]
    lines += ["| Subgroup | Value | N | Weighted accuracy |", "|---|---|---|---|"]
    d["was_outlier_b"] = as_bool(d["was_outlier"])
    for col in ("language_group", "bloc", "assembly", "was_outlier_b"):
        for value, g in d.groupby(col):
            gw = g["weight"].astype(float)
            lines.append(f"| {col} | {value} | {len(g)} | {(gw * g.correct).sum() / gw.sum():.1%} |")
    lines += ["", "Per-class precision (within predicted class, Wilson 95% CI) and weighted recall:", "",
              "| Class | Predicted N | Precision [95% CI] | Recall |", "|---|---|---|---|"]
    for label, g in d.groupby("predicted_label"):
        k = int(g.correct.sum())
        lo, hi = wilson(k, len(g))
        rec = weighted_pr(d["topic_label"] == label, d["predicted_label"] == label, w)["recall"]
        lines.append(f"| {label} | {len(g)} | {k / len(g):.0%} [{lo:.0%}, {hi:.0%}] | {rec:.0%} |")
    return lines + [""]


def score_procedural(key: pd.DataFrame, gold: pd.DataFrame) -> list[str]:
    d = key.join(gold, how="inner")
    lines = ["## Procedural events", "",
             "| Tag | Model value | Coded N | Human positives | Precision | Recall | F1 |", "|---|---|---|---|---|---|---|"]
    for tag in TAGS:
        sub = d.dropna(subset=[tag])
        truth = sub[tag].astype(str).str.lower().eq("yes")
        w = sub["weight"].astype(float)
        for variant in sorted({"rule", DOWNSTREAM[tag]}):
            r = weighted_pr(truth, as_bool(sub[f"{tag}_{variant}"]), w)
            lines.append(f"| {tag} | {variant} | {r['n']} | {r['n_true']} | {r['precision']:.1%} | {r['recall']:.1%} | {r['f1']:.3f} |")
    return lines + [""]


def score_attribution(key: pd.DataFrame, gold: pd.DataFrame) -> list[str]:
    d = key.join(gold, how="inner").dropna(subset=["chair_intervened"])
    d["chair"] = d["chair_intervened"].astype(str).str.lower().eq("yes")
    d["directed"] = d["chair"] & d["directed_at_speaker"].astype(str).str.lower().eq("yes")
    lines = ["## Chair attribution of H1 intervention windows", "",
             "| Bloc | Event type | N | Chair intervened | Chair intervention directed at speaker |", "|---|---|---|---|---|"]
    for (bloc, event), g in d.groupby(["bloc", "event_type"]):
        lines.append(f"| {bloc} | {event} | {len(g)} | {g.chair.mean():.0%} | {g.directed.mean():.0%} |")
    w = d["weight"].astype(float)
    for bloc, g in d.groupby("bloc"):
        gw = g["weight"].astype(float)
        lines.append(f"| {bloc} | all (weighted) | {len(g)} | {(gw * g.chair).sum() / gw.sum():.0%} | "
                     f"{(gw * g.directed).sum() / gw.sum():.0%} |")
    lines.append(f"\nWeighted share of H1 'interventions' that are chair interventions directed at the speaker: "
                 f"{(w * d.directed).sum() / w.sum():.0%}.")
    return lines + [""]


def score_pq_sensitive(key: pd.DataFrame, gold: pd.DataFrame) -> list[str]:
    d = key.join(gold, how="inner").dropna(subset=["sensitive"])
    truth = d["sensitive"].astype(str).str.lower().eq("yes")
    w = d["weight"].astype(float)
    lines = ["## H2 sensitivity lexicons", "", f"Coded PQs: {len(d)}; weighted share judged sensitive: "
             f"{(w * truth).sum() / w.sum():.1%}.", "", "| Definition | Precision | Recall | F1 |", "|---|---|---|---|"]
    for col in ("sensitive_core", "sensitive_extended", "sensitive_topic"):
        r = weighted_pr(truth, as_bool(d[col]), w)
        lines.append(f"| {col} | {r['precision']:.1%} | {r['recall']:.1%} | {r['f1']:.3f} |")
    return lines + [""]


def score_pq_status(key: pd.DataFrame, gold: pd.DataFrame) -> list[str]:
    d = key.join(gold, how="inner").dropna(subset=["status"])
    table = pd.crosstab(d["stratum"], d["status"].astype(str))
    accuracy = (d["stratum"] == d["status"].astype(str)).mean()
    return ["## PQ transfer/withdrawal parser", "", f"Coded items: {len(d)}; parser agrees with annotator on "
            f"{accuracy:.1%}.", "", "Rows: parser; columns: annotator.", "", table.to_markdown(), ""]


def run(annotator_paths: list[Path], adjudicated_path: Path | None, key_path: Path, out_path: Path) -> None:
    key = pd.read_parquet(key_path)
    annotators = [read_annotations(p) for p in annotator_paths]
    adjudicated = read_annotations(adjudicated_path) if adjudicated_path else None

    lines = ["# Annotation Validation Report", "",
             f"Annotators: {len(annotators)}" + ("; adjudicated file supplied." if adjudicated else "."), "",
             "## Inter-annotator agreement (double-coded rows)", "",
             "| Task | Field | N | Agreement | Cohen's kappa |", "|---|---|---|---|---|"]
    if len(annotators) >= 2:
        for task, spec in TASKS.items():
            fields = [f for f in spec["fields"] if f != "notes"]
            for r in agreement(annotators[0][task], annotators[1][task], fields):
                lines.append(f"| {task} | {r['field']} | {r['n']} | {r['agreement']:.1%} | {r['kappa']:.3f} |")
    lines.append("")

    scorers = {"topic": score_topic, "procedural": score_procedural, "attribution": score_attribution,
               "pq_sensitive": score_pq_sensitive, "pq_status": score_pq_status}
    for task, scorer in scorers.items():
        fields = [f for f in TASKS[task]["fields"] if f != "notes"]
        gold = gold_labels(annotators, adjudicated, task, fields)
        task_key = key[key["task"] == task].set_index("item_id")
        lines += scorer(task_key, gold)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotators", nargs="+", type=Path, required=True)
    parser.add_argument("--adjudicated", type=Path)
    parser.add_argument("--key", type=Path, default=Path("data/validation/annotation_key.parquet"))
    parser.add_argument("--out", type=Path, default=Path("data/validation/annotation_validation_report.md"))
    args = parser.parse_args()
    run(args.annotators, args.adjudicated, args.key, args.out)
