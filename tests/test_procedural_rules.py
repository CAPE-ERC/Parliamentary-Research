import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from procedural_layer.rules import tag_utterance


def test_interruption_stage_direction():
    tags = tag_utterance("(Interruptions)", is_stage_direction=True, role=None)
    assert tags["interruption"] is True
    assert tags["chair_ruling"] is False
    assert tags["withdrawal_request"] is False


def test_withdrawn_stage_direction():
    tags = tag_utterance("(Withdrawn)", is_stage_direction=True, role=None)
    assert tags["withdrawal_request"] is True
    assert tags["interruption"] is False


def test_withdrawal_dialogue_real_example():
    tags = tag_utterance("Withdraw! Withdraw that word!", is_stage_direction=False, role="speaker")
    assert tags["withdrawal_request"] is True
    assert tags["chair_ruling"] is True  # "withdraw" is also a chair-ruling keyword


def test_so_citation_real_example():
    text = (
        "Mr Deputy Speaker, Sir, I am pained to say this but I believe this is an "
        "abuse of supplementary questions and I will explain why. Standing Order 26 "
        "is very clear."
    )
    tags = tag_utterance(text, is_stage_direction=False, role="unclassified_mp")
    assert tags["so_citation"] is True


def test_pnq_transfer_real_example():
    text = (
        "Hon. Members, the Table has been advised that PQ B/378 will be replied by "
        "the hon. Minister of Finance, Economic Planning and Development"
    )
    tags = tag_utterance(text, is_stage_direction=False, role="speaker")
    assert tags["pnq_transfer"] is True


def test_pnq_transfer_would_variant():
    """Found via LSTM disagreement review - the corpus also uses 'would be
    replied by', not just 'will', and the original regex missed it."""
    text = "Hon. Members, the Table has been advised that PQ B/892 would be replied by the hon. Vice-Prime Minister"
    tags = tag_utterance(text, is_stage_direction=False, role="speaker")
    assert tags["pnq_transfer"] is True


def test_chair_ruling_requires_chair_role():
    text = "Order! I will ask you to come to the motion of tonight."
    chair_tags = tag_utterance(text, is_stage_direction=False, role="speaker")
    mp_tags = tag_utterance(text, is_stage_direction=False, role="unclassified_mp")
    assert chair_tags["chair_ruling"] is True
    assert mp_tags["chair_ruling"] is False


def test_generic_transfer_is_not_pnq_transfer():
    """Guards against the false-positive pattern found during grounding -
    plain 'transfer' (funds, staff, assets) is not a PNQ reassignment."""
    text = "Under item 28211 Transfers to Non-Profit Institutions, there are transfers to various Cooperatives."
    tags = tag_utterance(text, is_stage_direction=False, role="minister")
    assert tags["pnq_transfer"] is False


def test_no_tags_for_ordinary_policy_text():
    text = "Government will increase housing supply and home ownership for the economically disadvantaged."
    tags = tag_utterance(text, is_stage_direction=False, role="prime_minister")
    assert not any(tags.values())


def test_multiple_tags_can_co_occur():
    """A chair citing a Standing Order while ordering a withdrawal is
    so_citation + chair_ruling + withdrawal_request simultaneously."""
    text = "Under Standing Order 34, I order you to withdraw that word immediately."
    tags = tag_utterance(text, is_stage_direction=False, role="speaker")
    assert tags["so_citation"] is True
    assert tags["chair_ruling"] is True
    assert tags["withdrawal_request"] is True
