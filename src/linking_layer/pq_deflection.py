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

Announcements are parsed clause by clause. A Speaker announcement typically
mixes transfers ("PQ B/362 will be replied by the hon. Minister of ...") with
withdrawals ("PQs B/334 and B/341 have been withdrawn") and then calls the
next question, whose own "(No. B/xxx)" header follows in the same record. An
earlier version flagged every B/ number in such a record as transferred,
which counted withdrawn questions and the next question's header as
transfers; each number is now assigned to the clause it belongs to, and the
destination minister of each transfer is kept.

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
TRANSFER_RE = re.compile(
    r"(?:will|would) be replied by|(?:has|have|had) been transferred to(?= (?:the )?(?:Rt\.? )?hon)", re.IGNORECASE
)
WITHDRAWN_RE = re.compile(r"withdrawn", re.IGNORECASE)
PQNUM_RE = re.compile(r"B/(\d+)")
NEXT_QUESTION_RE = re.compile(r"\(No\.\s*B/\d+\)")
# A "following" list ("the following PQs will be replied by X: B/18 ..., B/34 ...")
# names its questions after the clause; the list runs to the next sentence
# that starts a new announcement.
LIST_END_RE = re.compile(r"\.\s+(?=PQ|The Table|Hon|I |Members)|!")
TITLE_PREFIX_RE = re.compile(r"^(?:the\s+)?(?:Rt\.?\s*)?(?:hon\.?\s*)?(?:Dr\.?\s*)?", re.IGNORECASE)
DESTINATION_END_RE = re.compile(
    r";|\s+at the end\b|,\s*time permitting|\bPQs?\b|\bHon\. Members\b|\bThe Table\b|!|:|\.(?=\s+[A-Z])(?<!hon\.)(?<!Ag\.)(?<!Dr\.)(?<!Rt\.)"
)


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


def _destination(text: str, start: int) -> str:
    tail = re.sub(r"\s+", " ", text[start:start + 200]).strip()
    tail = TITLE_PREFIX_RE.sub("", tail).strip()
    cut = DESTINATION_END_RE.search(tail)
    return (tail[: cut.start()] if cut else tail).strip(" .,")


def parse_announcement(text: str) -> list[tuple[str, str, str | None]]:
    """(pq_num, status, destination) for every PQ number in one Speaker
    announcement, where status is 'transferred' or 'withdrawn'. Numbers are
    assigned to the clause they belong to: normally the next transfer or
    withdrawal clause, or the preceding clause when it introduces a
    "following" list. Text from the next question's own header onward is
    ignored."""
    header = NEXT_QUESTION_RE.search(text)
    text = text[: header.start()] if header else text

    markers = []
    for m in TRANSFER_RE.finditer(text):
        markers.append((m.start(), m.end(), "transferred", _destination(text, m.end())))
    for m in WITHDRAWN_RE.finditer(text):
        markers.append((m.start(), m.end(), "withdrawn", None))
    markers.sort()

    lists = []
    for start, end, status, destination in markers:
        if re.search(r"following", text[max(0, start - 80):start], re.IGNORECASE):
            first = PQNUM_RE.search(text, end)
            if first is None:
                continue
            stop = LIST_END_RE.search(text, first.end())
            lists.append((end, stop.start() if stop else len(text), status, destination))

    results = []
    for m in PQNUM_RE.finditer(text):
        assigned = next(((st, d) for lo, hi, st, d in lists if lo <= m.start() < hi), None)
        if assigned is None:
            assigned = next(((st, d) for start, _, st, d in markers if start >= m.end()), None)
        if assigned is not None:
            results.append((m.group(1), assigned[0], assigned[1]))
    return results


def parse_announcements(utterances: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct (debate_id, pq_num) named in a transfer or
    withdrawal announcement. A question named in both is a transfer."""
    rows = []
    mask = utterances["text"].str.contains(TRANSFER_RE, regex=True, na=False) | utterances["text"].str.contains(
        WITHDRAWN_RE, regex=True, na=False
    )
    for r in utterances[mask].itertuples(index=False):
        for num, status, destination in parse_announcement(str(r.text)):
            rows.append({"debate_id": r.debate_id, "pq_num": num, "status": status, "destination": destination})
    df = pd.DataFrame(rows, columns=["debate_id", "pq_num", "status", "destination"])
    df["_order"] = df["status"].map({"transferred": 0, "withdrawn": 1})
    return df.sort_values("_order").drop_duplicates(["debate_id", "pq_num"]).drop(columns="_order").reset_index(drop=True)


def extract_transferred_pq_nums(utterances: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct (debate_id, pq_num) transferred to another
    minister, with the destination minister."""
    announced = parse_announcements(utterances)
    transferred = announced[announced["status"] == "transferred"]
    return transferred[["debate_id", "pq_num", "destination"]].reset_index(drop=True)


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
    announced = parse_announcements(utterances)
    transferred = announced[announced["status"] == "transferred"]

    headers = resolve_pq_party(headers, registry)
    headers = attach_topic(headers, utterances, topics)

    headers = headers.merge(announced, on=["debate_id", "pq_num"], how="left")
    headers["transferred"] = headers["status"].eq("transferred")
    headers["withdrawn"] = headers["status"].eq("withdrawn")
    headers = headers.drop(columns="status")

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
    print(f"Withdrawn PQs: {panel['withdrawn'].sum()}")
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
