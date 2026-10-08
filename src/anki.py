"""Anki export — build an importable ``.apkg`` deck from extracted words."""

from __future__ import annotations

import html
import zlib
from pathlib import Path
from typing import Iterable, Mapping

import genanki

# Fixed IDs: Anki identifies a note type by its ID, so changing it would create a
# duplicate "PodcastCard" note type on every import.
_MODEL_ID = 1607392319

_CSS = """
.card { font-family: "Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", sans-serif;
        font-size: 18px; text-align: center; color: #222; background: #fff; }
.word { font-size: 56px; margin: 12px 0; }
.pinyin { font-size: 24px; color: #5c7cfa; }
.definition { font-size: 20px; margin: 10px 0; }
.sentence { font-size: 22px; margin: 14px 0; }
.sentence b { color: #d9480f; }
.more { font-size: 16px; color: #666; }
.meta { font-size: 13px; color: #888; margin-top: 16px; }
"""

MODEL = genanki.Model(
    _MODEL_ID,
    "PodcastCard Vocabulary",
    fields=[
        {"name": "Word"},
        {"name": "Pinyin"},
        {"name": "Definition"},
        {"name": "Sentence"},
        {"name": "MoreSentences"},
        {"name": "HSK"},
        {"name": "Source"},
    ],
    templates=[
        {
            "name": "Recognition",
            "qfmt": '<div class="word">{{Word}}</div><div class="sentence">{{Sentence}}</div>',
            "afmt": (
                "{{FrontSide}}<hr id=answer>"
                '<div class="pinyin">{{Pinyin}}</div>'
                '<div class="definition">{{Definition}}</div>'
                '<div class="more">{{MoreSentences}}</div>'
                '<div class="meta">{{HSK}} · {{Source}}</div>'
            ),
        }
    ],
    css=_CSS,
)


def _pick_sentences(word: str, contexts: list[str]) -> tuple[str, list[str]]:
    """Choose the best example sentence and up to two more.

    Prefers the shortest sentence that is long enough to give real context.
    """
    if not contexts:
        return "", []
    min_len = max(6, len(word) + 3)
    usable = [c for c in contexts if len(c) >= min_len] or sorted(contexts, key=len, reverse=True)
    best = min(usable, key=len)
    rest = [c for c in contexts if c != best][:2]
    return best, rest


def _highlight(sentence: str, word: str) -> str:
    return html.escape(sentence).replace(html.escape(word), f"<b>{html.escape(word)}</b>")


def _note(word: Mapping, source: str) -> genanki.Note:
    text = word["word"]
    sentence, more = _pick_sentences(text, list(word.get("contexts") or []))
    level = word.get("hsk_level", 0)
    hsk_label = f"HSK {level}" if level else "Not on HSK list"
    return genanki.Note(
        model=MODEL,
        fields=[
            html.escape(text),
            html.escape(word.get("pinyin", "")),
            html.escape(word.get("definition", "")),
            _highlight(sentence, text),
            "<br>".join(_highlight(s, text) for s in more),
            hsk_label,
            html.escape(source),
        ],
        # Same word => same note, so re-importing another episode doesn't duplicate it.
        guid=genanki.guid_for("podcastcard", text),
        tags=["podcastcard", f"HSK{level}" if level else "HSK-unlisted"],
    )


def build_package(words: Iterable[Mapping], deck_name: str, source: str = "") -> genanki.Package:
    """Build a deck of one note per word (dicts with word/pinyin/definition/hsk_level/contexts)."""
    deck = genanki.Deck(zlib.crc32(deck_name.encode("utf-8")) | (1 << 30), deck_name)
    for word in words:
        deck.add_note(_note(word, source or deck_name))
    return genanki.Package(deck)


def write_apkg(words: Iterable[Mapping], path: str | Path, deck_name: str, source: str = "") -> int:
    """Write an ``.apkg`` file; returns the number of notes it contains."""
    package = build_package(words, deck_name, source)
    package.write_to_file(str(path))
    return len(package.decks[0].notes)
