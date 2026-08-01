"""Resolves party / government-opposition status for unclassified_mp
utterances using a real, researched speaker registry (data/external/
speaker_registry.csv) - see data/processed/party_resolution_report.md for
sourcing and the PMSD mid-term coalition-switch correction this depends on.

Matching is surname-based (Hansard's own convention), scoped to the debate's
Assembly term to avoid cross-term collisions, and time-aware for government/
opposition status (a party's alignment can change mid-term - see
party_alignment.csv). Surname collisions within the same term (multiple
registry entries with the same key) are left unresolved rather than guessed.
"""

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ASSEMBLY_TO_TERM = {"SIXTH": "6th", "SEVENTH": "7th", "EIGHTH": "8th"}

MONTH_RE = re.compile(
    r"(\d{1,2})\s+(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s+(\d{4})",
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    """Strip accents and lowercase, so registry keys need not exactly match
    the corpus's accented Unicode (or vice versa)."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return stripped.lower().strip()


def extract_candidate_surnames(speaker_raw: str) -> list[str]:
    """Returns candidate surname keys to try against the registry, longest
    (most specific) first - handles compound surnames like 'Ameer Meea' that
    Layer 0's single-last-token extraction would reduce to just 'Meea'."""
    name = speaker_raw
    paren = re.search(r"\(([^)]+)\)\s*$", name)
    if paren:
        name = paren.group(1)
    name = re.sub(r"^(Mr\.?|Mrs\.?|Ms\.?|Dr\.?|The)\s+", "", name).strip()
    tokens = [normalize(t.strip(".")) for t in name.split()]
    if not tokens:
        return []

    candidates = []
    if len(tokens) >= 2:
        candidates.append(" ".join(tokens[-2:]))
    candidates.append(tokens[-1])
    return candidates


def parse_sitting_date(sitting_date: str | None) -> pd.Timestamp | None:
    if not sitting_date:
        return None
    m = MONTH_RE.search(sitting_date)
    if not m:
        return None
    day, month, year = m.groups()
    try:
        return pd.to_datetime(f"{day} {month} {year}", format="%d %B %Y")
    except ValueError:
        return None


@dataclass
class Registry:
    # (assembly_term, surname_key) -> list of (full_name, party) - list because
    # collisions (same surname, same term, different people) must stay visible.
    by_term_surname: dict[tuple[str, str], list[tuple[str, str]]]
    # (party, assembly_term) -> list of (gov_or_opp, effective_from, effective_to)
    alignment: dict[tuple[str, str], list[tuple[str, pd.Timestamp | None, pd.Timestamp | None]]]


def load_registry(external_dir: Path) -> Registry:
    members = pd.read_csv(external_dir / "speaker_registry.csv")
    alignment_df = pd.read_csv(external_dir / "party_alignment.csv")

    by_term_surname: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for row in members.itertuples(index=False):
        key = (row.assembly_term, normalize(row.surname_key))
        by_term_surname.setdefault(key, []).append((row.full_name, row.party))

    alignment: dict[tuple[str, str], list] = {}
    for row in alignment_df.itertuples(index=False):
        key = (row.party, row.assembly_term)
        eff_from = pd.to_datetime(row.effective_from) if pd.notna(row.effective_from) else None
        eff_to = pd.to_datetime(row.effective_to) if pd.notna(row.effective_to) else None
        alignment.setdefault(key, []).append((row.gov_or_opp, eff_from, eff_to))

    return Registry(by_term_surname, alignment)


def resolve_gov_opp(registry: Registry, party: str, assembly_term: str, sitting_date: pd.Timestamp | None) -> str | None:
    windows = registry.alignment.get((party, assembly_term))
    if not windows:
        return None
    if len(windows) == 1:
        return windows[0][0]
    if sitting_date is None:
        return None  # ambiguous without a date when the party switched sides mid-term
    for gov_or_opp, eff_from, eff_to in windows:
        after_start = eff_from is None or sitting_date >= eff_from
        before_end = eff_to is None or sitting_date < eff_to
        if after_start and before_end:
            return gov_or_opp
    return None


def resolve_utterance(
    registry: Registry, speaker_raw: str, assembly: str, sitting_date_text: str | None
) -> dict:
    term = ASSEMBLY_TO_TERM.get(assembly)
    result = {"resolved_party": None, "resolved_gov_opp": None, "match_method": "no_match"}
    if not term or not speaker_raw:
        return result

    candidates = extract_candidate_surnames(speaker_raw)
    for candidate in candidates:
        matches = registry.by_term_surname.get((term, candidate))
        if not matches:
            continue
        if len(matches) > 1:
            result["match_method"] = "ambiguous_collision"
            return result
        full_name, party = matches[0]
        sitting_date = parse_sitting_date(sitting_date_text)
        gov_opp = resolve_gov_opp(registry, party, term, sitting_date)
        return {
            "resolved_party": party,
            "resolved_gov_opp": gov_opp,
            "match_method": "matched_2word" if " " in candidate else "matched_1word",
        }
    return result


def run(processed_dir: Path, external_dir: Path, years: list[str] | None = None) -> None:
    registry = load_registry(external_dir)
    df = pd.read_parquet(processed_dir / "utterances.parquet")
    if years:
        df = df[df["year"].isin(years)]
    unresolved = df[(df["role"] == "unclassified_mp") & (~df["is_stage_direction"])].copy()

    print(f"Resolving party for {len(unresolved)} unclassified_mp utterances...")
    results = [
        resolve_utterance(registry, row.speaker_raw, row.assembly, row.sitting_date)
        for row in unresolved.itertuples(index=False)
    ]
    results_df = pd.DataFrame(results)
    out = pd.concat([unresolved[["debate_id", "seq_index"]].reset_index(drop=True), results_df], axis=1)

    out.to_parquet(processed_dir / "speaker_party_resolved.parquet", index=False)

    method_counts = out["match_method"].value_counts()
    govopp_counts = out["resolved_gov_opp"].value_counts(dropna=False)
    print("\nMatch method counts:")
    print(method_counts)
    print("\nResolved government/opposition counts:")
    print(govopp_counts)

    with open(processed_dir / "party_resolution_report.md", "w", encoding="utf-8") as f:
        f.write("# Speaker Party Resolution Report\n\n")
        f.write(f"- Unclassified MP utterances scanned: {len(unresolved)}\n")
        f.write(f"- Resolved to a party: {(out['resolved_party'].notna()).sum()}\n")
        f.write(f"- Resolved to government/opposition: {(out['resolved_gov_opp'].notna()).sum()}\n")
        f.write(f"- Ambiguous (surname collision within same term): "
                f"{(out['match_method'] == 'ambiguous_collision').sum()}\n")
        f.write(f"- No match found: {(out['match_method'] == 'no_match').sum()}\n\n")
        f.write("## Match method breakdown\n\n")
        f.write(method_counts.to_frame("count").to_markdown())
        f.write("\n\n## Resolved government/opposition breakdown\n\n")
        f.write(govopp_counts.to_frame("count").to_markdown())
        f.write(
            "\n\nSee the plan/report for the PMSD mid-term coalition-switch correction "
            "(19 Dec 2016) and the low-confidence Rodrigues-party (RPO) alignment "
            "assumption documented in data/external/party_alignment.csv.\n"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--external-dir", default="data/external")
    parser.add_argument("--years", nargs="*", default=None)
    args = parser.parse_args()

    run(Path(args.processed_dir), Path(args.external_dir), args.years)
