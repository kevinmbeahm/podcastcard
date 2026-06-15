"""HSK lookup module — loads word→level mapping and provides pinyin lookup."""

from __future__ import annotations

import json
from pathlib import Path

from pypinyin import lazy_pinyin, Style

# Locate data file relative to this file's package root (src/../data/)
_DATA_FILE = Path(__file__).parent.parent / "data" / "hsk_words.json"


def _load_hsk_words() -> dict[str, int]:
    with _DATA_FILE.open(encoding="utf-8") as fh:
        return json.load(fh)


HSK_WORDS: dict[str, int] = _load_hsk_words()


def get_hsk_level(word: str) -> int:
    """Return HSK level (1-6) for *word*, or 0 if not in the HSK word list."""
    return HSK_WORDS.get(word, 0)


def get_pinyin(word: str) -> str:
    """Return space-separated pinyin (with tone marks) for *word*."""
    return " ".join(lazy_pinyin(word, style=Style.TONE))
