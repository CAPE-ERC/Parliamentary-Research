"""Front-matter parsing: debate header, Cabinet list, and Principal Officers.

Hansard PDFs print this metadata on the first few pages of every debate. It's
extracted per-debate (rather than sourced from one static list) because the
Cabinet and chair officers change across the 2015-2025 corpus as elections and
reshuffles occur.
"""

import re
from dataclasses import dataclass, field

HONORIFIC_PREFIXES = re.compile(
    r"^(Dr\.\s+the\s+Hon\.|Hon\.\s+Mrs|Hon\.\s+Ms|Hon\.\s+Dr\.|Hon\.\s+Sir|Hon\.)\s*",
    re.IGNORECASE,
)
POST_NOMINAL = re.compile(r",?\s*\b([A-Z]{2,6})\b\.?$")

PORTFOLIO_TRIGGERS = [
    "Prime Minister",
    "Deputy Prime Minister",
    "Vice-Prime Minister",
    "Minister of",
    "Minister for",
    "Attorney General",
]

OFFICER_ROLES = [
    "Deputy Chairperson of Committees",
    "Clerk of the National Assembly",
    "Deputy Clerk",
    "Clerk Assistant",
    "Hansard Editor",
    "Serjeant-at-Arms",
    "Deputy Speaker",
    "Madam Speaker",
    "Mr Speaker",
]


@dataclass
class DebateMetadata:
    debate_number: str | None = None
    assembly: str | None = None
    session: str | None = None
    sitting_date: str | None = None
    cabinet: dict[str, str] = field(default_factory=dict)  # surname_key -> portfolio
    officers: dict[str, str] = field(default_factory=dict)  # role_key -> surname


def _surname_from_name(name: str) -> str | None:
    """Best-effort surname extraction, e.g. 'Sir Anerood Jugnauth, GCSK' -> 'jugnauth'."""
    name = HONORIFIC_PREFIXES.sub("", name).strip()
    name = re.sub(r"^Sir\s+", "", name)
    name_part = name.split(",")[0].strip()
    tokens = [t for t in name_part.split() if t]
    if not tokens:
        return None
    return tokens[-1].lower().strip(".")


def parse_header(page_text: str) -> tuple[str | None, str | None, str | None, str | None]:
    """Parse debate number, assembly, session, sitting date from the title page."""
    debate_match = re.search(r"No\.\s*(\d+[A-Za-z]?)\s*of\s*(\d{4})", page_text)
    debate_number = debate_match.group(0) if debate_match else None

    assembly_match = re.search(r"([A-Z]+)\s+NATIONAL ASSEMBLY", page_text)
    assembly = assembly_match.group(1) if assembly_match else None

    session_match = re.search(r"([A-Z]+)\s+SESSION", page_text)
    session = session_match.group(1) if session_match else None

    date_match = re.search(
        r"([A-Z]+DAY)\s+(\d{1,2})\s+([A-Z]+)\s+(\d{4})", page_text
    )
    sitting_date = date_match.group(0) if date_match else None

    return debate_number, assembly, session, sitting_date


def parse_cabinet(front_matter_text: str) -> dict[str, str]:
    """Parse the THE CABINET block into {surname_key: portfolio_text}."""
    start = front_matter_text.find("THE CABINET")
    end = front_matter_text.find("PRINCIPAL OFFICERS")
    if start == -1:
        return {}
    block = front_matter_text[start : end if end != -1 else start + 4000]
    lines = [l.strip() for l in block.split("\n") if l.strip()]

    cabinet: dict[str, str] = {}
    current_name, current_portfolio = None, []
    for line in lines[1:]:  # skip "THE CABINET" line itself
        if line.startswith("(Formed by"):
            continue
        if line.startswith("Hon.") or line.startswith("Dr. the Hon."):
            if current_name:
                surname = _surname_from_name(current_name)
                if surname:
                    cabinet[surname] = " ".join(current_portfolio).strip()
            trigger_idx = None
            for trig in PORTFOLIO_TRIGGERS:
                idx = line.find(trig)
                if idx != -1 and (trigger_idx is None or idx < trigger_idx):
                    trigger_idx = idx
            if trigger_idx is not None:
                current_name = line[:trigger_idx].strip().rstrip(",")
                current_portfolio = [line[trigger_idx:].strip()]
            else:
                current_name = line
                current_portfolio = []
        else:
            current_portfolio.append(line)

    if current_name:
        surname = _surname_from_name(current_name)
        if surname:
            cabinet[surname] = " ".join(current_portfolio).strip()

    return cabinet


def parse_officers(front_matter_text: str) -> dict[str, str]:
    """Parse the PRINCIPAL OFFICERS AND OFFICIALS block into {role_key: surname}."""
    start = front_matter_text.find("PRINCIPAL OFFICERS")
    if start == -1:
        return {}
    end = front_matter_text.find("\n", start)
    block = front_matter_text[start + (end - start if end != -1 else 0) : start + 2000]

    officers: dict[str, str] = {}
    for role in OFFICER_ROLES:
        idx = block.find(role)
        if idx == -1:
            continue
        rest = block[idx + len(role) :]
        next_idxs = [
            block.find(r, idx + len(role))
            for r in OFFICER_ROLES
            if block.find(r, idx + len(role)) != -1
        ]
        cutoff = min(next_idxs) if next_idxs else idx + len(role) + 200
        entry = block[idx + len(role) : cutoff].strip()
        surname = entry.split(",")[0].strip().lower()
        role_key = "speaker" if role in ("Madam Speaker", "Mr Speaker") else role.lower().replace(" ", "_")
        if surname:
            officers[role_key] = surname

    return officers


def parse_metadata(pages: list[str]) -> DebateMetadata:
    """Parse all front-matter metadata from a debate's extracted pages."""
    front_matter_text = "\n".join(pages[:8])
    header_text = pages[0] if pages else ""

    debate_number, assembly, session, sitting_date = parse_header(header_text)
    cabinet = parse_cabinet(front_matter_text)
    officers = parse_officers(front_matter_text)

    return DebateMetadata(
        debate_number=debate_number,
        assembly=assembly,
        session=session,
        sitting_date=sitting_date,
        cabinet=cabinet,
        officers=officers,
    )
