"""Transcript helpers — tokenise for the reader UI and format for export."""

from __future__ import annotations

import re
from typing import Iterable

from .dictionary import get_definition
from .extract import Segment
from .hsk import get_hsk_level, get_pinyin
from .segment import tokenize  # noqa: F401  (re-exported)

# Basic ideographs, Extension A (rare characters), compatibility forms, Extensions B-G
_CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿\U00020000-\U0003134f]")


def has_cjk(token: str) -> bool:
    """True if *token* contains at least one Chinese character."""
    return bool(_CJK_RE.search(token))


def lexicon_entry(token: str) -> dict:
    """What the reader shows for a word without asking the dictionary again."""
    return {
        "pinyin": get_pinyin(token),
        "hsk_level": get_hsk_level(token),
        "definition": get_definition(token),
    }


def annotate(segments: Iterable[Segment]) -> tuple[list[dict], dict[str, dict]]:
    """Tokenise *segments* and build a lexicon for every Chinese token in them.

    Returns ``(segments, lexicon)`` where each segment is
    ``{start, end, text, tokens}`` and the lexicon maps a token to
    ``{pinyin, hsk_level, definition}``. Unlike vocabulary extraction,
    nothing is filtered out, so every word in the transcript can be looked up.
    """
    out: list[dict] = []
    lexicon: dict[str, dict] = {}
    for seg in segments:
        tokens = tokenize(seg.text)
        for token in tokens:
            if has_cjk(token) and token not in lexicon:
                lexicon[token] = lexicon_entry(token)
        out.append({"start": seg.start, "end": seg.end, "text": seg.text, "tokens": tokens})
    return out, lexicon


def _timestamp(seconds: float, sep: str) -> str:
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def format_vtt(segments: Iterable[Segment]) -> str:
    """WebVTT subtitle file (also usable as a time-coded transcript)."""
    lines = ["WEBVTT", ""]
    for seg in segments:
        lines += [f"{_timestamp(seg.start, '.')} --> {_timestamp(seg.end, '.')}", seg.text, ""]
    return "\n".join(lines)


def format_text(segments: Iterable[Segment]) -> str:
    """Plain-text transcript with a ``[mm:ss]`` marker per segment."""
    lines = []
    for seg in segments:
        m, s = divmod(int(seg.start), 60)
        lines.append(f"[{m:02d}:{s:02d}] {seg.text}")
    return "\n".join(lines) + "\n"
