"""Per-utterance language identification: English / French / Kreol Morisien.

Deviates from the milestone doc's "fastText language detection": fastText
requires a C++ build toolchain that isn't available on this machine (no
Microsoft Visual C++ Build Tools), so we use `langdetect` (pure Python)
instead. Neither fastText's lid.176 model nor langdetect has a Kreol Morisien
class, so a keyword heuristic is layered on top as an override - Kreol is
under-resourced and gets misclassified as French/English otherwise. This is a
known limitation: verify the language distribution in the corpus quality
report rather than trusting it blindly, especially for short utterances.
"""

import re

from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 0  # deterministic results

KREOL_KEYWORDS = {
    "mo", "to", "li", "nou", "zot", "ou", "pou", "ena", "pa", "dan", "sa",
    "kot", "kouma", "kifer", "kot", "zafer", "bizin", "gagne", "fer",
    "seki", "sanla", "komie", "kitfoi", "labous",
}

_WORD_RE = re.compile(r"[a-zà-ÿ]+", re.IGNORECASE)


def _kreol_keyword_hits(text: str) -> int:
    words = {w.lower() for w in _WORD_RE.findall(text)}
    return len(words & KREOL_KEYWORDS)


def detect_language(text: str, min_reliable_chars: int = 20) -> str:
    """Return 'en', 'fr', 'mfe' (Kreol Morisien), or 'unknown'.

    langdetect is unreliable below ~20 chars (short interjections like "Yes."
    get misclassified as Tagalog, Welsh, etc.). Below that threshold we
    default to 'en' - the corpus's dominant language - unless Kreol keywords
    are present, since a wrong guess among ~50 languages is worse than
    assuming the majority language for text too short to meaningfully judge.
    """
    text = text.strip()
    if not text:
        return "unknown"

    hits = _kreol_keyword_hits(text)
    if hits >= 2:
        return "mfe"

    if len(text) < min_reliable_chars:
        return "mfe" if hits >= 1 else "en"

    try:
        detected = detect(text)
    except LangDetectException:
        return "unknown"

    if detected == "fr" and hits >= 1:
        return "mfe"

    return detected
