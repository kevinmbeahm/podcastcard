"""Word segmentation for Simplified, Traditional or mixed Chinese text."""

from __future__ import annotations

import jieba

from .dictionary import simplify_characters


def tokenize(text: str) -> list[str]:
    """Split *text* into words. The tokens always join back to *text* exactly.

    jieba's word list is Simplified, so on Traditional text it falls back to guessing and
    chops words into single characters. Each Traditional character has exactly one
    Simplified counterpart, so the text is segmented in Simplified and the original text is
    cut at the same positions: 詞彙 stays 詞彙, as one word.
    """
    simplified = simplify_characters(text)
    tokens, position = [], 0
    for token in jieba.lcut(simplified, cut_all=False):
        tokens.append(text[position : position + len(token)])
        position += len(token)
    return tokens
