"""Builds the human-annotation validation sample for the text-derived measures.

Topic test labels in the paper derive from reviewed clusters and the
procedural classifier is scored against rule-generated silver labels, so
neither establishes accuracy against independent annotation. This module
draws a stratified sample and writes a blind annotation workbook (no model
output shown) plus a separate key file holding model labels, strata and
sampling weights for scoring (see score_annotations.py).

Tasks (one sheet each):
  topic        - policy domain of an utterance (or non_policy). Stratified by
                 predicted label, bloc, language and term; predicted
                 non_policy is sampled too so policy recall can be estimated.
  procedural   - which of the five procedural events an utterance contains.
                 Rule positives for each tag are oversampled, with rule/BiLSTM
                 disagreements and random negatives.
  attribution  - for H1 intervention windows: did the chair intervene, and
                 was the intervention directed at the preceding speaker?
  pq_sensitive - is a Parliamentary Question politically sensitive?
                 Stratified by the lexicon flags used in H2.
  pq_status    - is a PQ named in a Speaker announcement transferred,
                 withdrawn, or neither? Checks the clause-level parser.

A share of each sheet is marked for double coding to estimate inter-annotator
agreement.

Usage:
    python -m validation.annotation_sample
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.worksheet.datavalidation import DataValidation

from linking_layer.build_panel import WINDOW, resolve_final_party
from linking_layer.h1_reanalysis import build_utterance_frame
from topic_layer.train_classifier import NON_POLICY_LABEL

SEED = 20261007
SIZES = {"topic_policy_per_label": 15, "topic_non_policy": 250, "procedural_per_tag": 80,
         "procedural_disagreements_per_tag": 20, "procedural_negatives": 150, "attribution": 200,
         "pq_sensitive_per_stratum": 100, "pq_status_per_stratum": 70}
DOUBLE_CODE_SHARE = 0.2
TAGS = ["interruption", "withdrawal_request", "chair_ruling", "so_citation", "pnq_transfer"]
LANGUAGES = {"en": "English", "fr": "French", "mfe": "Kreol Morisien"}
CONTEXT = 2


def language_group(code: str | None) -> str:
    return LANGUAGES.get(code, "Other/indeterminate")


def stratified(df: pd.DataFrame, by: list[str], n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Up to n rows spread as evenly as possible across the strata in `by`."""
    groups = list(df.groupby(by, dropna=False))
    per = max(1, n // max(1, len(groups)))
    picks = [g.sample(min(len(g), per), random_state=int(rng.integers(1e9))) for _, g in groups]
    out = pd.concat(picks)
    if len(out) < n:
        rest = df.drop(out.index)
        out = pd.concat([out, rest.sample(min(len(rest), n - len(out)), random_state=int(rng.integers(1e9)))])
    return out.head(n)


def with_weights(sample: pd.DataFrame, population: pd.DataFrame, stratum: str) -> pd.DataFrame:
    """Inverse-probability weights so estimates generalise to the population."""
    pop = population[stratum].value_counts()
    got = sample[stratum].value_counts()
    return sample.assign(weight=sample[stratum].map(pop / got))


def context_text(utts: pd.DataFrame, debate_id: str, seq_index: int, before: int, after: int) -> str:
    rows = utts[(utts["debate_id"] == debate_id) & utts["seq_index"].between(seq_index - before, seq_index + after)]
    parts = []
    for r in rows.itertuples():
        marker = ">>> " if r.seq_index == seq_index else ""
        if isinstance(r.speaker_raw, str):
            speaker = r.speaker_raw
        else:
            speaker = "(stage direction)" if r.is_stage_direction else "(speaker not identified)"
        parts.append(f"{marker}[{speaker}] {str(r.text)[:600]}")
    return "\n".join(parts)


def load(processed_dir: Path) -> dict:
    utts = pd.read_parquet(processed_dir / "utterances.parquet",
                           columns=["debate_id", "seq_index", "assembly", "speaker_raw", "role", "language", "text",
                                    "is_stage_direction"])
    party = pd.read_parquet(processed_dir / "speaker_party_resolved.parquet")
    utts = resolve_final_party(utts, party)
    labels = pd.read_parquet(processed_dir / "utterance_policy_labels_two_stage.parquet")
    confidence = pd.read_parquet(processed_dir / "topic_assignment_confidence.parquet",
                                 columns=["debate_id", "seq_index", "was_outlier", "low_confidence"])
    tags = pd.read_parquet(processed_dir / "procedural_tags_final.parquet")
    return {"utts": utts, "labels": labels, "confidence": confidence, "tags": tags}


def topic_task(d: dict, rng) -> pd.DataFrame:
    df = d["labels"].merge(d["utts"], on=["debate_id", "seq_index"]).merge(d["confidence"], on=["debate_id", "seq_index"], how="left")
    df["bloc"] = df["speaker_party_final"].fillna("chair/unresolved")
    df["language_group"] = df["language"].map(language_group)
    df["stratum"] = np.where(df["predicted_label"] == NON_POLICY_LABEL, NON_POLICY_LABEL, df["predicted_label"])
    policy = df[df["predicted_label"] != NON_POLICY_LABEL]
    picks = [stratified(g, ["bloc", "language_group", "assembly"], SIZES["topic_policy_per_label"], rng)
             for _, g in policy.groupby("predicted_label")]
    non_policy = stratified(df[df["predicted_label"] == NON_POLICY_LABEL], ["bloc", "language_group", "assembly"],
                            SIZES["topic_non_policy"], rng)
    sample = with_weights(pd.concat(picks + [non_policy]), df, "stratum")
    sample["display_text"] = sample["text"].str[:1500]
    return sample


def procedural_task(d: dict, rng) -> pd.DataFrame:
    df = d["tags"].merge(d["utts"], on=["debate_id", "seq_index"])
    df["language_group"] = df["language"].map(language_group)
    picks = []
    for tag in TAGS:
        rule = df[df[f"{tag}_rule"] == 1]
        picks.append(stratified(rule, ["assembly", "language_group"], SIZES["procedural_per_tag"], rng).assign(stratum=f"{tag}_rule"))
        disagree = df[df[f"{tag}_rule"] != df[f"{tag}_lstm"]]
        if len(disagree):
            picks.append(disagree.sample(min(len(disagree), SIZES["procedural_disagreements_per_tag"]),
                                         random_state=int(rng.integers(1e9))).assign(stratum=f"{tag}_disagreement"))
    any_rule = df[[f"{t}_rule" for t in TAGS]].sum(axis=1) > 0
    picks.append(stratified(df[~any_rule], ["assembly", "language_group"], SIZES["procedural_negatives"], rng)
                 .assign(stratum="rule_negative"))
    sample = pd.concat(picks)
    sample = sample[~sample.index.duplicated()]
    df["stratum"] = "rule_negative"
    for tag in TAGS:
        df.loc[df[f"{tag}_rule"] != df[f"{tag}_lstm"], "stratum"] = f"{tag}_disagreement"
    for tag in TAGS:
        df.loc[df[f"{tag}_rule"] == 1, "stratum"] = f"{tag}_rule"
    sample = with_weights(sample, df, "stratum")
    sample["display_text"] = [context_text(d["utts"], r.debate_id, r.seq_index, 1, 1) for r in sample.itertuples()]
    return sample


def attribution_task(d: dict, processed_dir: Path, rng) -> pd.DataFrame:
    frame = build_utterance_frame(processed_dir).rename(columns={"sitting": "debate_id"})
    intervened = frame[frame["y_any_w3"]].copy()
    intervened["bloc"] = np.where(intervened["opp"] == 1, "opposition", "government")
    intervened["event_type"] = np.where(intervened["y_ruling_w3"], "ruling_in_window", "interruption_only")
    sample = stratified(intervened, ["bloc", "event_type", "term"], SIZES["attribution"], rng)
    sample = sample.assign(stratum=sample["bloc"] + "|" + sample["event_type"])
    intervened["stratum"] = intervened["bloc"] + "|" + intervened["event_type"]
    sample = with_weights(sample, intervened, "stratum")
    sample["display_text"] = [context_text(d["utts"], r.debate_id, r.seq_index, 0, WINDOW) for r in sample.itertuples()]
    return sample


def pq_tasks(processed_dir: Path, d: dict, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    from linking_layer.h2_reanalysis import question_text

    frame = pd.read_parquet(processed_dir / "h2_frame.parquet")
    frame["question_text"] = question_text(frame, d["utts"])
    frame["stratum"] = np.select(
        [frame["sensitive_core"], frame["sensitive_extended"]], ["core_lexicon", "extended_only"], "no_lexicon"
    )
    sensitive = pd.concat([g.sample(min(len(g), SIZES["pq_sensitive_per_stratum"]), random_state=int(rng.integers(1e9)))
                           for _, g in frame.groupby("stratum")])
    sensitive = with_weights(sensitive, frame, "stratum")
    sensitive["display_text"] = sensitive["question_text"].str[:1500]

    panel = pd.read_parquet(processed_dir / "pq_deflection_panel.parquet")
    panel["stratum"] = np.select([panel["transferred"], panel["withdrawn"]], ["transferred", "withdrawn"], "neither")
    # "Neither" PQs are only informative when their number appears in an announcement.
    from linking_layer.pq_deflection import TRANSFER_RE, WITHDRAWN_RE
    ann = d["utts"][d["utts"]["text"].str.contains(TRANSFER_RE, na=False) | d["utts"]["text"].str.contains(WITHDRAWN_RE, na=False)]
    ann_by_debate = ann.groupby("debate_id")["text"].apply(list).to_dict()

    def snippet(row) -> str | None:
        for text in ann_by_debate.get(row.debate_id, []):
            for token in (f"B/{row.pq_num},", f"B/{row.pq_num} ", f"B/{row.pq_num}.", f"B/{row.pq_num})", f"B/{row.pq_num};"):
                i = text.find(token)
                if i >= 0:
                    return text[max(0, i - 500):i + 400]
        return None

    candidates = panel.copy()
    candidates["display_text"] = candidates.apply(snippet, axis=1)
    candidates = candidates[candidates["display_text"].notna()]
    status = pd.concat([g.sample(min(len(g), SIZES["pq_status_per_stratum"]), random_state=int(rng.integers(1e9)))
                        for _, g in candidates.groupby("stratum")])
    status = with_weights(status, candidates, "stratum")
    return sensitive, status


TASKS = {
    "topic": {
        "instructions": "Read the utterance. Choose the single policy domain it is mainly about, or non_policy if it "
                        "is procedural, ceremonial, or not about a policy matter.",
        "fields": {"topic_label": "LIST:topics", "confident": "LIST:yes,no", "notes": None},
    },
    "procedural": {
        "instructions": "The marked line (>>>) is the record being coded; neighbours are context. Mark each event "
                        "that the marked record itself contains.",
        "fields": {f"{t}": "LIST:yes,no" for t in TAGS} | {"notes": None},
    },
    "attribution": {
        "instructions": "The first line (>>>) is a member's contribution; the following lines are the next records. "
                        "Did the presiding chair intervene (ruling, call to order, request to withdraw) in these "
                        "records, and if so was it directed at the member in the first line? An '(Interruptions)' "
                        "stage direction alone is not a chair intervention.",
        "fields": {"chair_intervened": "LIST:yes,no", "directed_at_speaker": "LIST:yes,no,unclear", "notes": None},
    },
    "pq_sensitive": {
        "instructions": "Is this Parliamentary Question politically sensitive, i.e. does it concern alleged "
                        "wrongdoing, corruption, misuse of public funds or office, irregular procurement or "
                        "appointments, or matters likely to embarrass the government? Code from the question text only.",
        "fields": {"sensitive": "LIST:yes,no", "notes": None},
    },
    "pq_status": {
        "instructions": "The excerpt is a Speaker announcement. For the PQ number shown in the pq_number column, "
                        "is that question transferred to another minister, withdrawn, or neither?",
        "fields": {"status": "LIST:transferred,withdrawn,neither", "notes": None},
    },
}


def write_workbook(path: Path, samples: dict, topic_labels: list[str], rng) -> pd.DataFrame:
    wb = Workbook()
    intro = wb.active
    intro.title = "README"
    intro["A1"] = "Annotation workbook - Mauritius National Assembly Hansard"
    intro["A1"].font = Font(bold=True, size=13)
    notes = [
        "Each sheet is one task. Fill in the coloured columns only; do not edit item_id or text.",
        "Work independently. Do not consult model output or other annotators' sheets.",
        f"Rows with double_code = yes are coded by both annotators ({int(DOUBLE_CODE_SHARE * 100)}% of each sheet).",
        "Use the notes column for anything ambiguous.",
        "",
    ] + [f"{name}: {spec['instructions']}" for name, spec in TASKS.items()]
    for i, line in enumerate(notes, start=3):
        intro.cell(row=i, column=1, value=line).alignment = Alignment(wrap_text=True)
    intro.column_dimensions["A"].width = 140

    lists = wb.create_sheet("lists")
    for i, label in enumerate(topic_labels, start=1):
        lists.cell(row=i, column=1, value=label)
    lists.sheet_state = "hidden"

    keys = []
    for name, sample in samples.items():
        sample = sample.sample(frac=1, random_state=int(rng.integers(1e9))).reset_index(drop=True)
        sample["item_id"] = [f"{name}-{i + 1:04d}" for i in range(len(sample))]
        sample["double_code"] = np.where(rng.random(len(sample)) < DOUBLE_CODE_SHARE, "yes", "no")
        ws = wb.create_sheet(name)
        base_cols = ["item_id", "double_code"] + (["pq_number"] if name == "pq_status" else []) + ["text"]
        fields = TASKS[name]["fields"]
        headers = base_cols + list(fields)
        for c, h in enumerate(headers, start=1):
            ws.cell(row=1, column=c, value=h).font = Font(bold=True)
        for r, row in enumerate(sample.itertuples(), start=2):
            ws.cell(row=r, column=1, value=row.item_id)
            ws.cell(row=r, column=2, value=row.double_code)
            col = 3
            if name == "pq_status":
                ws.cell(row=r, column=col, value=f"B/{row.pq_num}")
                col += 1
            ws.cell(row=r, column=col, value=row.display_text).alignment = Alignment(wrap_text=True, vertical="top")
        text_col = len(base_cols)
        ws.column_dimensions[ws.cell(row=1, column=text_col).column_letter].width = 110
        for offset, (field, spec) in enumerate(fields.items(), start=text_col + 1):
            letter = ws.cell(row=1, column=offset).column_letter
            ws.column_dimensions[letter].width = 22
            if spec is None:
                continue
            source = f"=lists!$A$1:$A${len(topic_labels)}" if spec == "LIST:topics" else f'"{spec[5:]}"'
            dv = DataValidation(type="list", formula1=source, allow_blank=True)
            ws.add_data_validation(dv)
            dv.add(f"{letter}2:{letter}{len(sample) + 1}")
        ws.freeze_panes = "A2"

        key_cols = [c for c in sample.columns if c not in ("text", "display_text", "question_text", "question_body")]
        keys.append(sample[key_cols].assign(task=name))
    wb.save(path)
    return pd.concat(keys, ignore_index=True)


def run(processed_dir: Path, out_dir: Path) -> None:
    rng = np.random.default_rng(SEED)
    d = load(processed_dir)
    print("Sampling...")
    samples = {
        "topic": topic_task(d, rng),
        "procedural": procedural_task(d, rng),
        "attribution": attribution_task(d, processed_dir, rng),
    }
    samples["pq_sensitive"], samples["pq_status"] = pq_tasks(processed_dir, d, rng)
    topic_labels = sorted(l for l in d["labels"]["predicted_label"].unique() if l != NON_POLICY_LABEL) + [NON_POLICY_LABEL]

    out_dir.mkdir(parents=True, exist_ok=True)
    key = write_workbook(out_dir / "annotation_workbook.xlsx", samples, topic_labels, rng)
    for col in key.columns:
        if key[col].dtype == object:
            key[col] = key[col].astype(str)
    key.to_parquet(out_dir / "annotation_key.parquet", index=False)
    print({k: len(v) for k, v in samples.items()})
    print(f"Wrote {out_dir / 'annotation_workbook.xlsx'} and the scoring key (keep the key away from annotators).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--out-dir", default="data/validation")
    args = parser.parse_args()
    run(Path(args.processed_dir), Path(args.out_dir))
