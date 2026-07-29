"""Splits a debate's page text into speaker-attributed utterances.

Hansard turns are printed as `SPEAKER_TAG: text...`, with wrapped
continuation lines carrying no tag. Stage directions (sitting
suspended/resumed, "(Mr Speaker in the Chair)", etc.) are emitted as
separate, unattributed records rather than folded into a speaker's turn.
"""

import re
from dataclasses import dataclass

PAGE_NUMBER_LINE = re.compile(r"^\d{1,4}$")

SHORT_TAG = re.compile(r"^(Mr\.?|Mrs\.?|Ms\.?|Dr\.?|Madam)\s+([A-Za-zÀ-ÿ'\-. ]{1,40}?):\s*(.*)$")
LONG_TAG = re.compile(r"^The\s+([A-Za-zÀ-ÿ'\-.,()0-9 ]{1,120}?):\s*(.*)$")
LONG_TAG_KEYWORDS = (
    "Minister",
    "Speaker",
    "Leader of the Opposition",
    "Chairperson",
    "President",
    "Whip",
)

STAGE_DIRECTION_PATTERNS = [
    re.compile(r"^At\s+\d{1,2}[.:]\d{2}\s*(a\.m\.|p\.m\.|hrs)", re.IGNORECASE),
    re.compile(r"^On resuming", re.IGNORECASE),
    re.compile(r"the sitting (was|is) (suspended|adjourned|resumed)", re.IGNORECASE),
    re.compile(r"^\(.{2,80}\)$"),
    re.compile(r"^The National Anthem was played", re.IGNORECASE),
    re.compile(r"^The Assembly met", re.IGNORECASE),
]


@dataclass
class Utterance:
    seq_index: int
    speaker_raw: str | None
    text: str
    is_stage_direction: bool


def _match_speaker_tag(line: str) -> tuple[str, str] | None:
    m = SHORT_TAG.match(line)
    if m:
        speaker = f"{m.group(1)} {m.group(2)}".strip()
        return speaker, m.group(3)

    m = LONG_TAG.match(line)
    if m and any(kw in m.group(1) for kw in LONG_TAG_KEYWORDS):
        speaker = f"The {m.group(1)}".strip()
        return speaker, m.group(2)

    return None


def _is_stage_direction(line: str) -> bool:
    return any(p.search(line) for p in STAGE_DIRECTION_PATTERNS)


def segment_utterances(pages: list[str], start_page: int = 0) -> list[Utterance]:
    """Segment debate pages (from start_page onward) into utterance records.

    Long-form tags (e.g. "The Vice-Prime Minister, Minister of Energy and
    Public Utilities (Mr I.\\nCollendavelloo): ...") routinely wrap across two
    extracted lines, splitting the name from its closing "):". A one-line
    lookahead join handles this without broadly merging unrelated lines.
    """
    lines: list[str] = []
    for page in pages[start_page:]:
        for raw_line in page.split("\n"):
            line = raw_line.strip()
            if line and not PAGE_NUMBER_LINE.match(line):
                lines.append(line)

    utterances: list[Utterance] = []
    current_speaker: str | None = None
    current_text: list[str] = []
    seq = 0

    def flush() -> None:
        nonlocal current_speaker, current_text, seq
        text = " ".join(t for t in current_text if t).strip()
        if text:
            utterances.append(Utterance(seq, current_speaker, text, False))
            seq += 1
        current_speaker = None
        current_text = []

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]

        if _is_stage_direction(line):
            flush()
            utterances.append(Utterance(seq, None, line, True))
            seq += 1
            i += 1
            continue

        tag_match = _match_speaker_tag(line)
        consumed_next = False
        if not tag_match and line.startswith("The ") and i + 1 < n:
            combined = f"{line} {lines[i + 1]}"
            tag_match = _match_speaker_tag(combined)
            consumed_next = tag_match is not None

        if tag_match:
            flush()
            current_speaker, first_chunk = tag_match
            current_text = [first_chunk]
            i += 2 if consumed_next else 1
            continue

        current_text.append(line)
        i += 1

    flush()
    return utterances
