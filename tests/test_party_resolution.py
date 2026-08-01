import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from linking_layer.party_resolution import (
    Registry,
    extract_candidate_surnames,
    normalize,
    resolve_gov_opp,
    resolve_utterance,
)


def test_normalize_strips_accents_and_lowercases():
    assert normalize("Bérenger") == "berenger"
    assert normalize("  Duval  ") == "duval"


def test_extract_candidate_surnames_prefers_compound_first():
    candidates = extract_candidate_surnames("Mr Ameer Meea")
    assert candidates[0] == "ameer meea"
    assert candidates[1] == "meea"


def test_extract_candidate_surnames_strips_honorific_and_parens():
    candidates = extract_candidate_surnames("The Vice-Prime Minister (Mr I. Collendavelloo)")
    assert "collendavelloo" in candidates


def _make_registry():
    by_term = {
        ("6th", "berenger"): [("Paul Berenger", "MMM")],
        ("6th", "duval"): [("Adrien Duval", "PMSD"), ("Xavier-Luc Duval", "PMSD")],  # real collision
        ("6th", "perraud"): [("Aurore Perraud", "PMSD")],
    }
    alignment = {
        ("MMM", "6th"): [("opposition", None, None)],
        ("PMSD", "6th"): [
            ("government", pd.Timestamp("2014-12-10"), pd.Timestamp("2016-12-19")),
            ("opposition", pd.Timestamp("2016-12-19"), pd.Timestamp("2019-11-07")),
        ],
    }
    return Registry(by_term, alignment)


def test_unambiguous_surname_resolves():
    registry = _make_registry()
    result = resolve_utterance(registry, "Mr Berenger", "SIXTH", "TUESDAY 10 FEBRUARY 2015")
    assert result["resolved_party"] == "MMM"
    assert result["resolved_gov_opp"] == "opposition"
    assert result["match_method"] == "matched_1word"


def test_colliding_surname_is_flagged_not_guessed():
    registry = _make_registry()
    result = resolve_utterance(registry, "Hon. Duval", "SIXTH", "TUESDAY 10 FEBRUARY 2015")
    assert result["match_method"] == "ambiguous_collision"
    assert result["resolved_party"] is None


def test_time_aware_party_switch_before_and_after():
    registry = _make_registry()
    before = resolve_gov_opp(registry, "PMSD", "6th", pd.Timestamp("2015-01-01"))
    after = resolve_gov_opp(registry, "PMSD", "6th", pd.Timestamp("2018-01-01"))
    assert before == "government"
    assert after == "opposition"


def test_time_aware_switch_without_date_is_ambiguous():
    registry = _make_registry()
    assert resolve_gov_opp(registry, "PMSD", "6th", None) is None


def test_unknown_term_returns_no_match():
    registry = _make_registry()
    result = resolve_utterance(registry, "Mr Berenger", "EIGHTH", "TUESDAY 10 FEBRUARY 2025")
    assert result["match_method"] == "no_match"
