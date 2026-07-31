"""Rule-based tagging for the five procedural-power phenomena.

Patterns are grounded in real matches pulled from the corpus, not assumed from
the tag names - e.g. generic "transfer" and plain "PNQ" keyword searches were
mostly false positives (fund/staff transfers, routine question headers); the
Table's formulaic "will be replied by" reassignment announcement is the actual
reliable signal for pnq_transfer.

Tags are NOT mutually exclusive: a chair citing a Standing Order while
ordering a withdrawal is so_citation + chair_ruling + withdrawal_request at
once. tag_utterance() returns a dict of independent booleans, not a class.
"""

import re

SO_CITATION_RE = re.compile(r"Standing Order\s*\d+", re.IGNORECASE)
WITHDRAW_RE = re.compile(r"\bwithdraw(n|ing)?\b", re.IGNORECASE)
PNQ_TRANSFER_RE = re.compile(r"(will|would) be replied by", re.IGNORECASE)

CHAIR_ROLES = {"speaker", "deputy_speaker", "deputy_chairperson_of_committees"}

CHAIR_RULING_KEYWORDS = re.compile(
    r"\b("
    r"order|ruling|i rule|not in order|out of order|overrule|over-rule|"
    r"sustained|disallow|suspend|please be seated|withdraw|"
    r"point of order|standing order|continue|silence|quiet"
    r")\b",
    re.IGNORECASE,
)

INTERRUPTION_STAGE_TEXT = "(interruptions)"
WITHDRAWN_STAGE_TEXT = "(withdrawn)"


def is_interruption(text: str, is_stage_direction: bool) -> bool:
    return is_stage_direction and text.strip().lower() == INTERRUPTION_STAGE_TEXT


def is_withdrawal_request(text: str, is_stage_direction: bool) -> bool:
    if is_stage_direction:
        return text.strip().lower() == WITHDRAWN_STAGE_TEXT
    return bool(WITHDRAW_RE.search(text))


def is_so_citation(text: str, is_stage_direction: bool) -> bool:
    if is_stage_direction:
        return False
    return bool(SO_CITATION_RE.search(text))


def is_pnq_transfer(text: str, is_stage_direction: bool) -> bool:
    if is_stage_direction:
        return False
    return bool(PNQ_TRANSFER_RE.search(text))


def is_chair_ruling(text: str, is_stage_direction: bool, role: str | None) -> bool:
    if is_stage_direction or role not in CHAIR_ROLES:
        return False
    return bool(CHAIR_RULING_KEYWORDS.search(text))


def tag_utterance(text: str, is_stage_direction: bool, role: str | None) -> dict[str, bool]:
    """Apply all five rules to one utterance."""
    return {
        "chair_ruling": is_chair_ruling(text, is_stage_direction, role),
        "interruption": is_interruption(text, is_stage_direction),
        "withdrawal_request": is_withdrawal_request(text, is_stage_direction),
        "so_citation": is_so_citation(text, is_stage_direction),
        "pnq_transfer": is_pnq_transfer(text, is_stage_direction),
    }
