"""H2 (question deflection) rebuilt at individual-PQ granularity.

The original H2 test (see regression.py's h2_pnq_deflection and
linking_layer_report.md) counted 226 `pnq_transfer`-tagged utterances split
across 50 topics - most topic cells had 0-2 events, too sparse to test
anything. Reading a sample of those utterances found why: 84% are the
Speaker's routine end-of-Question-Time announcement ("Time over! ... PQ B/757
will be replied by ...") and each one bundles a median of 3 straggler PQ
numbers together - a scheduling artifact, not 226 independent deflection
decisions.

Every Parliamentary Question is printed with a "(No. B/xxx) <Name>
(<Constituency>)" header in the same convention Layer 0 already parses for
roles (see preprocessing/roles.py) - 12,426 distinct PQs are recoverable this
way corpus-wide, giving each one a known asker. 97% of the PQ numbers
referenced inside transfer announcements match a known header in the same
debate. That lets H2 be tested as intended: transferred ~ asker's party,
across the full PQ population, rather than a sparse per-topic breakdown.

Usage:
    python -m linking_layer.pq_deflection
"""

import argparse
import re
from pathlib import Path

import pandas as pd

from linking_layer.party_resolution import (
    ASSEMBLY_TO_TERM,
    extract_candidate_surnames,
    load_registry,
    parse_sitting_date,
    resolve_gov_opp,
    Registry,
)

HEADER_RE = re.compile(
    r"\(No\.\s*B/(\d+)\)\s*((?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[^(]+?)\s*\(([^)]+)\)"
)
TRANSFER_RE = re.compile(r"(?:will|would) be replied by", re.IGNORECASE)
PQNUM_RE = re.compile(r"B/(\d+)")


def extract_pq_headers(utterances: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct (debate_id, pq_num) PQ header found in the
    corpus, with the asker's raw name/constituency and the seq_index to
    resolve topic from (the header's own row if it carries the question
    itself, otherwise the immediately following row - see module docstring)."""
    rows = []
    for r in utterances.itertuples(index=False):
        for m in HEADER_RE.finditer(str(r.text)):
            rows.append(
                {
                    "debate_id": r.debate_id,
                    "pq_num": m.group(1),
                    "asker_raw": m.group(2).strip(),
                    "constituency_raw": m.group(3).strip(),
                    "header_seq_index": r.seq_index,
                    "header_is_stage_direction": r.is_stage_direction,
                    "assembly": r.assembly,
                    "sitting_date": r.sitting_date,
                }
            )
    df = pd.DataFrame(rows)
    return df.drop_duplicates(subset=["debate_id", "pq_num"]).reset_index(drop=True)


def extract_transferred_pq_nums(utterances: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct (debate_id, pq_num) that appears inside a
    "will/would be replied by" transfer announcement - unbundling the batched
    Speaker announcements into individual transferred questions."""
    rows = []
    mask = utterances["text"].str.contains(TRANSFER_RE, regex=True, na=False)
    for r in utterances[mask].itertuples(index=False):
        for num in PQNUM_RE.findall(str(r.text)):
            rows.append({"debate_id": r.debate_id, "pq_num": num})
    df = pd.DataFrame(rows, columns=["debate_id", "pq_num"])
    return df.drop_duplicates().reset_index(drop=True)


def resolve_pq_party(headers: pd.DataFrame, registry: Registry) -> pd.DataFrame:
    df = headers.copy()
    resolved_party = []
    resolved_gov_opp = []
    match_method = []
    for r in df.itertuples(index=False):
        term = ASSEMBLY_TO_TERM.get(r.assembly)
        party = gov_opp = None
        method = "no_match"
        if term:
            for candidate in extract_candidate_surnames(r.asker_raw):
                matches = registry.by_term_surname.get((term, candidate))
                if not matches:
                    continue
                if len(matches) > 1:
                    method = "ambiguous_collision"
                    break
                full_name, party = matches[0]
                sitting_date = parse_sitting_date(r.sitting_date)
                gov_opp = resolve_gov_opp(registry, party, term, sitting_date)
                method = "matched_2word" if " " in candidate else "matched_1word"
                break
        resolved_party.append(party)
        resolved_gov_opp.append(gov_opp)
        match_method.append(method)
    df["resolved_party"] = resolved_party
    df["resolved_gov_opp"] = resolved_gov_opp
    df["match_method"] = match_method
    return df


def attach_topic(headers: pd.DataFrame, utterances: pd.DataFrame, topics: pd.DataFrame) -> pd.DataFrame:
    """The question's own topic label: if the header carries the question
    text itself, use its own (debate_id, seq_index); if the header is a bare
    stage-direction line, the question text is the immediately following
    utterance in the same debate (verified against a sample - see module
    docstring)."""
    df = headers.copy()
    topic_lookup = topics.set_index(["debate_id", "seq_index"])["predicted_label"]

    next_seq = (
        utterances.sort_values(["debate_id", "seq_index"])
        .groupby("debate_id")["seq_index"]
        .shift(-1)
    )
    next_seq_lookup = utterances[["debate_id", "seq_index"]].copy()
    next_seq_lookup["next_seq_index"] = next_seq.values
    next_seq_lookup = next_seq_lookup.set_index(["debate_id", "seq_index"])["next_seq_index"]

    def lookup_topic(row) -> str | None:
        key = (row.debate_id, row.header_seq_index)
        if not row.header_is_stage_direction:
            return topic_lookup.get(key)
        next_idx = next_seq_lookup.get(key)
        if next_idx is None or pd.isna(next_idx):
            return None
        return topic_lookup.get((row.debate_id, next_idx))

    df["topic"] = df.apply(lookup_topic, axis=1)
    return df


def build_pq_panel(processed_dir: Path, external_dir: Path) -> pd.DataFrame:
    utterances = pd.read_parquet(processed_dir / "utterances.parquet")
    topics = pd.read_parquet(processed_dir / "utterance_policy_labels_two_stage.parquet")
    registry = load_registry(external_dir)

    headers = extract_pq_headers(utterances)
    transferred = extract_transferred_pq_nums(utterances)

    headers = resolve_pq_party(headers, registry)
    headers = attach_topic(headers, utterances, topics)

    transferred_keys = set(zip(transferred["debate_id"], transferred["pq_num"]))
    headers["transferred"] = [
        (d, n) in transferred_keys for d, n in zip(headers["debate_id"], headers["pq_num"])
    ]

    n_transfer_refs = len(transferred)
    n_matched = sum(1 for d, n in zip(transferred["debate_id"], transferred["pq_num"]) if (d, n) in set(zip(headers["debate_id"], headers["pq_num"])))
    headers.attrs["n_transfer_refs"] = n_transfer_refs
    headers.attrs["n_transfer_refs_matched"] = n_matched
    return headers


def run(processed_dir: Path, external_dir: Path) -> None:
    panel = build_pq_panel(processed_dir, external_dir)
    panel.to_parquet(processed_dir / "pq_deflection_panel.parquet", index=False)

    print(f"Distinct PQs found: {len(panel)}")
    print(f"Transfer references found: {panel.attrs['n_transfer_refs']} "
          f"({panel.attrs['n_transfer_refs_matched']} matched to a known header)")
    print(f"Transferred PQs (unbundled): {panel['transferred'].sum()}")
    print("\nAsker resolution:")
    print(panel["match_method"].value_counts())
    print("\nGov/opp distribution among resolved askers:")
    print(panel["resolved_gov_opp"].value_counts(dropna=False))
    print("\nTopic coverage:")
    print(f"{panel['topic'].notna().sum()} / {len(panel)} PQs have a resolved topic")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--external-dir", default="data/external")
    args = parser.parse_args()

    run(Path(args.processed_dir), Path(args.external_dir))
