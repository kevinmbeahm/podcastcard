"""Transcript helpers — tokenise for the reader UI and format for export."""

from __future__ import annotations

import re
from typing import Iterable

import jieba

from .dictionary import get_definition
from .extract import Segment
from .hsk import get_hsk_level, get_pinyin

_CJK_RE = re.compile(r"[一-鿿]")


def has_cjk(token: str) -> bool:
    """True if *token* contains at least one Chinese character."""
    return bool(_CJK_RE.search(token))


def tokenize(text: str) -> list[str]:
    """Split *text* into words. The tokens always join back to *text* exactly."""
    return jieba.lcut(text, cut_all=False)


def _entry(word: str) -> dict:
    entry = {
        "pinyin": get_pinyin(word),
        "hsk_level": get_hsk_level(word),
        "definition": get_definition(word),
    }
    if not entry["definition"] and len(word) > 1:
        # Unknown compound (names, slang): fall back to a per-character breakdown.
        entry["parts"] = [
            {"char": ch, "pinyin": get_pinyin(ch), "definition": get_definition(ch)}
            for ch in word
            if has_cjk(ch)
        ]
    return entry


def annotate(segments: Iterable[Segment]) -> tuple[list[dict], dict[str, dict]]:
    """Tokenise *segments* and build a lexicon for every Chinese token in them.

    Returns ``(segments, lexicon)`` where each segment is
    ``{start, end, text, tokens}`` and the lexicon maps a token to
    ``{pinyin, hsk_level, definition[, parts]}``. Unlike vocabulary extraction,
    nothing is filtered out, so every word in the transcript can be looked up.
    """
    out: list[dict] = []
    lexicon: dict[str, dict] = {}
    for seg in segments:
        tokens = tokenize(seg.text)
        for token in tokens:
            if has_cjk(token) and token not in lexicon:
                lexicon[token] = _entry(token)
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
