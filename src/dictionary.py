"""CC-CEDICT lookup — English definitions for Chinese words.

Data: ``data/cedict_ts.u8.gz`` (CC-CEDICT, CC BY-SA 4.0, https://cc-cedict.org).
The file is parsed lazily on first lookup.
"""

from __future__ import annotations

import gzip
import re
from functools import lru_cache
from pathlib import Path

from pypinyin import lazy_pinyin, Style

_DATA_FILE = Path(__file__).parent.parent / "data" / "cedict_ts.u8.gz"

# "Traditional Simplified [pin1 yin1] /def one/def two/"
_LINE_RE = re.compile(r"^(\S+) (\S+) \[([^\]]*)\] /(.*)/\s*$")

# Definitions that add noise on a flashcard (cross-references, proper-noun
# abbreviations, measure-word hints)
_NOISE_PREFIXES = (
    "CL:",
    "see ",
    "see also",
    "variant of",
    "old variant of",
    "also written",
    "erhua variant",
    "abbr. for",
    "surname",
    "Taiwan pr.",
)

_MAX_DEFINITIONS = 3

# simplified -> list of (numeric pinyin, [definitions])
_Entry = tuple[str, list[str]]


@lru_cache(maxsize=1)
def _load() -> dict[str, list[_Entry]]:
    entries: dict[str, list[_Entry]] = {}
    with gzip.open(_DATA_FILE, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            m = _LINE_RE.match(line)
            if not m:
                continue
            _trad, simp, pinyin, defs = m.groups()
            entries.setdefault(simp, []).append((pinyin, defs.split("/")))
    return entries


def _numeric_pinyin(word: str) -> str:
    """Pinyin in CEDICT's style (``ni3 hao3``) for matching heteronyms."""
    return " ".join(lazy_pinyin(word, style=Style.TONE3, neutral_tone_with_five=True)).lower()


def _clean(defs: list[str]) -> list[str]:
    return [d for d in defs if d and not d.startswith(_NOISE_PREFIXES)]


def get_definition(word: str) -> str:
    """Return a short English gloss for *word*, or ``""`` if not in CEDICT.

    Entries are ranked: exact reading match (CEDICT capitalises proper nouns,
    so ``xin1`` and ``Xin1`` are different), then case-insensitive reading
    match, then the rest. The first entry with a usable definition wins;
    cross-reference-only entries (``variant of ...``) are skipped.
    """
    entries = _load().get(word)
    if not entries:
        return ""
    wanted = _numeric_pinyin(word)
    ranked = sorted(
        entries,
        key=lambda e: (e[0] != wanted, e[0].lower() != wanted),  # stable sort
    )
    for _pinyin, defs in ranked:
        cleaned = _clean(defs)
        if cleaned:
            return "; ".join(cleaned[:_MAX_DEFINITIONS])
    return ""
