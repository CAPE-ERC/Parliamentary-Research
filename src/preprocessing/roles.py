"""Best-effort role/party derivation for a speaker tag, using that debate's
Cabinet + Officers metadata (see metadata.py).

Known limitation: matching is by surname only, so Cabinet members who share a
surname with another Cabinet or Officer entry (e.g. two unrelated MPs both
named "Jugnauth" or "Duval" in the same Assembly) cannot be disambiguated
here. Left as unclassified where the surname is genuinely ambiguous is not
implemented (last-write-wins in metadata.py); full-name matching would need a
speaker registry that doesn't exist yet - see data/external/speaker_registry_TEMPLATE.csv.
"""

import re
from dataclasses import dataclass

from .metadata import DebateMetadata

ROLE_PHRASES = {
    "prime_minister": re.compile(r"^(The\s+)?(Ag\.\s+)?Prime Minister$", re.IGNORECASE),
    "leader_of_opposition": re.compile(r"^The Leader of the Opposition", re.IGNORECASE),
    "speaker": re.compile(r"^(Mr|Madam)\s+Speaker$", re.IGNORECASE),
}


@dataclass
class SpeakerRole:
    role: str
    party: str | None


def _extract_surname(speaker_raw: str) -> str:
    name = speaker_raw
    # Strip a trailing "(Mr X)" parenthetical if present, e.g. long-form tags.
    paren = re.search(r"\(([^)]+)\)\s*$", name)
    if paren:
        name = paren.group(1)
    name = re.sub(r"^(Mr\.?|Mrs\.?|Ms\.?|Dr\.?|The)\s+", "", name).strip()
    tokens = name.split()
    return tokens[-1].lower().strip(".") if tokens else ""


def resolve_role(speaker_raw: str, metadata: DebateMetadata) -> SpeakerRole:
    stripped = speaker_raw.strip()

    for role_key, pattern in ROLE_PHRASES.items():
        if pattern.match(stripped):
            party = "government" if role_key == "prime_minister" else None
            if role_key == "speaker":
                party = "chair"
            if role_key == "leader_of_opposition":
                party = "opposition"
            return SpeakerRole(role_key, party)

    surname = _extract_surname(speaker_raw)
    if not surname:
        return SpeakerRole("unclassified_mp", None)

    if surname in metadata.cabinet:
        return SpeakerRole("minister", "government")

    for role_key, officer_surname in metadata.officers.items():
        if surname == officer_surname:
            return SpeakerRole(role_key, "chair")

    return SpeakerRole("unclassified_mp", None)
