import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from linking_layer.party_resolution import Registry
from linking_layer.pq_deflection import (
    attach_topic,
    extract_pq_headers,
    extract_transferred_pq_nums,
    resolve_pq_party,
)


def _utterances(rows):
    base = {
        "debate_id": "d1",
        "assembly": "EIGHTH",
        "sitting_date": "TUESDAY 10 FEBRUARY 2022",
        "is_stage_direction": False,
    }
    return pd.DataFrame([{**base, **r} for r in rows])


def test_extract_pq_headers_dedupes_and_captures_fields():
    utterances = _utterances(
        [
            {
                "seq_index": 0,
                "text": "(No. B/42) Mr A. Ameer Meea (Second Member for Port Louis Maritime & Port Louis East)",
                "is_stage_direction": True,
            },
            {"seq_index": 1, "text": "asked the Minister of Health whether he will state."},
            # duplicate header (e.g. reprinted) - should be deduped
            {
                "seq_index": 2,
                "text": "(No. B/42) Mr A. Ameer Meea (Second Member for Port Louis Maritime & Port Louis East)",
                "is_stage_direction": True,
            },
        ]
    )
    headers = extract_pq_headers(utterances)
    assert len(headers) == 1
    row = headers.iloc[0]
    assert row["pq_num"] == "42"
    assert row["asker_raw"] == "Mr A. Ameer Meea"
    assert row["constituency_raw"] == "Second Member for Port Louis Maritime & Port Louis East"
    assert row["header_seq_index"] == 0


def test_extract_transferred_pq_nums_unbundles_multiple_refs():
    utterances = _utterances(
        [
            {
                "seq_index": 0,
                "text": "Time over! PQ B/10 will be replied by the hon. Minister. PQ B/11 would be replied by the hon. Deputy Prime Minister.",
            },
        ]
    )
    transferred = extract_transferred_pq_nums(utterances)
    assert set(transferred["pq_num"]) == {"10", "11"}
    assert len(transferred) == 2


def test_resolve_pq_party_matches_and_flags_collision():
    registry = Registry(
        by_term_surname={
            ("8th", "ameer meea"): [("A. Ameer Meea", "MMM")],
            ("8th", "duval"): [("Adrien Duval", "PMSD"), ("Xavier-Luc Duval", "PMSD")],
        },
        alignment={("MMM", "8th"): [("opposition", None, None)]},
    )
    headers = pd.DataFrame(
        [
            {
                "debate_id": "d1",
                "pq_num": "1",
                "asker_raw": "Mr A. Ameer Meea",
                "assembly": "EIGHTH",
                "sitting_date": "TUESDAY 10 FEBRUARY 2022",
            },
            {
                "debate_id": "d1",
                "pq_num": "2",
                "asker_raw": "Mr Duval",
                "assembly": "EIGHTH",
                "sitting_date": "TUESDAY 10 FEBRUARY 2022",
            },
        ]
    )
    resolved = resolve_pq_party(headers, registry)
    matched = resolved[resolved["pq_num"] == "1"].iloc[0]
    assert matched["resolved_party"] == "MMM"
    assert matched["resolved_gov_opp"] == "opposition"
    assert matched["match_method"] == "matched_2word"

    collided = resolved[resolved["pq_num"] == "2"].iloc[0]
    assert pd.isna(collided["resolved_party"])
    assert collided["match_method"] == "ambiguous_collision"


def test_attach_topic_uses_own_row_or_next_row():
    utterances = pd.DataFrame(
        [
            # header is its own stage-direction row -> topic comes from the next row
            {"debate_id": "d1", "seq_index": 0, "is_stage_direction": True, "text": "(No. B/1) header"},
            {"debate_id": "d1", "seq_index": 1, "is_stage_direction": False, "text": "asked the Minister..."},
            # header is glued into the question utterance itself -> topic comes from its own row
            {"debate_id": "d1", "seq_index": 2, "is_stage_direction": False, "text": "(No. B/2) header asked..."},
        ]
    )
    topics = pd.DataFrame(
        [
            {"debate_id": "d1", "seq_index": 1, "predicted_label": "Healthcare"},
            {"debate_id": "d1", "seq_index": 2, "predicted_label": "Education"},
        ]
    )
    headers = pd.DataFrame(
        [
            {"debate_id": "d1", "pq_num": "1", "header_seq_index": 0, "header_is_stage_direction": True},
            {"debate_id": "d1", "pq_num": "2", "header_seq_index": 2, "header_is_stage_direction": False},
        ]
    )
    result = attach_topic(headers, utterances, topics)
    assert result[result["pq_num"] == "1"].iloc[0]["topic"] == "Healthcare"
    assert result[result["pq_num"] == "2"].iloc[0]["topic"] == "Education"
