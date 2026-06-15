"""Word extraction — tokenise transcript segments and collect vocabulary."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import jieba

from .hsk import get_hsk_level, get_pinyin

# Common single-character Mandarin function words to discard
_FUNCTION_CHARS: frozenset[str] = frozenset(
    "的了是在我你他她它们这那和也都有不就把被从到与及"
)

_PUNCT_RE = re.compile(r"^[\W\d_]+$", re.UNICODE)


def _is_noise(token: str) -> bool:
    """Return True if the token should be discarded."""
    token = token.strip()
    if not token:
        return True
    # Pure punctuation / numbers
    if _PUNCT_RE.match(token):
        return True
    # Single character that is a known function word
    if len(token) == 1 and token in _FUNCTION_CHARS:
        return True
    return False


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class WordOccurrence:
    word: str
    pinyin: str
    hsk_level: int
    contexts: list[str] = field(default_factory=list)


def extract_words(segments: list[Segment]) -> list[WordOccurrence]:
    """
    Tokenise each segment with jieba, deduplicate words, and return a
    list of WordOccurrence objects sorted by HSK level (unknown last).
    """
    # word -> list of context sentences
    word_contexts: dict[str, list[str]] = {}

    for seg in segments:
        sentence = seg.text.strip()
        if not sentence:
            continue
        tokens = jieba.lcut(sentence, cut_all=False)
        seen_in_sentence: set[str] = set()
        for token in tokens:
            if _is_noise(token):
                continue
            if token not in seen_in_sentence:
                seen_in_sentence.add(token)
                word_contexts.setdefault(token, [])
                word_contexts[token].append(sentence)

    occurrences: list[WordOccurrence] = []
    for word, contexts in word_contexts.items():
        occurrences.append(
            WordOccurrence(
                word=word,
                pinyin=get_pinyin(word),
                hsk_level=get_hsk_level(word),
                contexts=contexts,
            )
        )

    # Sort: known HSK levels first (ascending), then unknown (0) last
    occurrences.sort(key=lambda w: (w.hsk_level == 0, w.hsk_level))
    return occurrences
