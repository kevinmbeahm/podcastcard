"""Regenerate data/hsk_words.json from drkameleon/complete-hsk-vocabulary (MIT).

Usage:
    python scripts/build_hsk_words.py [path/to/complete.json]

Without an argument the dataset is downloaded from GitHub. Levels are taken
from the classic HSK 2.0 lists (``old-1`` .. ``old-6``). Words absent from
HSK 2.0 but present in HSK 3.0 levels 1-6 (``new-1`` .. ``new-6``) fall back to
their 3.0 level, so basics such as 说 / 天 / 山 are not reported as unknown.
Bare *characters* from the 3.0 fallback are only kept for levels 1-3: at levels 4-6 they are
mostly characters that live inside compounds (入, 于, 作, 公 ...), and the standards disagree about
their level (入 is 6 in HSK 3.0 but 4 in the 2025 revision), so they would show up as "HSK 6
words" that are really just parts of words. HSK 2.0 single-character entries are always kept.
Words in neither list are level 0 at lookup time and are not stored.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

SOURCE_URL = (
    "https://raw.githubusercontent.com/drkameleon/complete-hsk-vocabulary/"
    "main/complete.json"
)
OUT_FILE = Path(__file__).parent.parent / "data" / "hsk_words.json"

# HSK 3.0 single characters are only trusted as stand-alone words up to this level.
MAX_FALLBACK_CHARACTER_LEVEL = 3

# Genuine HSK 2.0 entries missing from the source dataset.
SUPPLEMENT: dict[str, int] = {"你好": 1}


def _load(argv: list[str]) -> list[dict]:
    if len(argv) > 1:
        return json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    with urllib.request.urlopen(SOURCE_URL, timeout=120) as resp:
        return json.load(resp)


def build(entries: list[dict]) -> dict[str, int]:
    hsk2: dict[str, int] = {}
    hsk3: dict[str, int] = {}
    for e in entries:
        word = e["simplified"]
        for tag in e["level"]:
            prefix, _, num = tag.partition("-")
            level = int(num)
            if prefix == "old":
                hsk2[word] = min(level, hsk2.get(word, 9))
            elif prefix == "new" and level <= 6:
                if len(word) == 1 and level > MAX_FALLBACK_CHARACTER_LEVEL:
                    continue
                hsk3[word] = min(level, hsk3.get(word, 9))
    return {**hsk3, **SUPPLEMENT, **hsk2}


def write(words: dict[str, int]) -> None:
    """One line per level keeps the file diff-friendly."""
    lines = []
    for level in range(1, 7):
        group = sorted(w for w, lv in words.items() if lv == level)
        lines.append(
            "  " + ", ".join(f"{json.dumps(w, ensure_ascii=False)}: {level}" for w in group)
        )
    OUT_FILE.write_text("{\n" + ",\n".join(lines) + "\n}\n", encoding="utf-8")


if __name__ == "__main__":
    words = build(_load(sys.argv))
    write(words)
    counts = {lv: sum(1 for x in words.values() if x == lv) for lv in range(1, 7)}
    print(f"wrote {len(words)} words to {OUT_FILE}: {counts}")
