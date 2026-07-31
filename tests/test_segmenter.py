import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preprocessing.segmenter import segment_utterances

SAMPLE_PAGES = [
    """7
ORAL ANSWER TO QUESTION
CEB - GENERATORS - TENDER
Mr Berenger: I had asked whether we can have the reasons why the Central
Procurement Board cancelled the tender.
Mr Collendavelloo: No, there has been no police enquiry on this matter. There
has been no decision on that.
(Interruptions)
Madam Speaker: Order!
The Vice-Prime Minister, Minister of Energy and Public Utilities (Mr I.
Collendavelloo): Let me clarify further on this point for the House.
At 11.36 a.m. the sitting was suspended.
On resuming at 1200 hrs in the Chamber.
The Prime Minister: I move that all business on today's Order Paper be set
aside."""
]


def test_short_tag_and_continuation():
    utts = segment_utterances(SAMPLE_PAGES)
    dialogue = [u for u in utts if not u.is_stage_direction]
    speakers = [u.speaker_raw for u in dialogue]
    assert "Mr Berenger" in speakers
    assert "Mr Collendavelloo" in speakers

    berenger = next(u for u in dialogue if u.speaker_raw == "Mr Berenger")
    assert "Central Procurement Board cancelled the tender" in berenger.text


def test_long_tag_with_role_keyword():
    utts = segment_utterances(SAMPLE_PAGES)
    dialogue = [u for u in utts if not u.is_stage_direction]
    long_tag = next(
        (u for u in dialogue if u.speaker_raw and "Vice-Prime Minister" in u.speaker_raw),
        None,
    )
    assert long_tag is not None
    assert "clarify further" in long_tag.text


def test_stage_directions_isolated_not_merged_into_speech():
    utts = segment_utterances(SAMPLE_PAGES)
    stage_texts = [u.text for u in utts if u.is_stage_direction]
    assert any("Interruptions" in t for t in stage_texts)
    assert any("sitting was suspended" in t for t in stage_texts)
    assert any("On resuming" in t for t in stage_texts)

    collendavelloo = next(
        u for u in utts if u.speaker_raw == "Mr Collendavelloo" and not u.is_stage_direction
    )
    assert "Interruptions" not in collendavelloo.text
    assert "Order!" not in collendavelloo.text


def test_prime_minister_short_role_tag():
    utts = segment_utterances(SAMPLE_PAGES)
    pm = next((u for u in utts if u.speaker_raw == "The Prime Minister"), None)
    assert pm is not None
    assert "business on today" in pm.text


def test_page_numbers_are_dropped():
    utts = segment_utterances(SAMPLE_PAGES)
    all_text = " ".join(u.text for u in utts)
    assert all_text.strip().split(" ")[0] != "7"
