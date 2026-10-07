"""Word segmentation of Simplified, Traditional and mixed text."""

import pytest

from src.extract import Segment, extract_words
from src.segment import tokenize


@pytest.mark.parametrize(
    "text",
    [
        "我們學習詞彙，也學習國家的歷史。",  # Traditional
        "我们学习词汇，也学习国家的历史。",  # Simplified
        "我們學習 hello 詞彙，我们学习。",  # mixed, with Latin text and spaces
        "",
    ],
)
def test_tokens_always_rejoin_to_the_original_text(text):
    assert "".join(tokenize(text)) == text


def test_traditional_words_are_not_chopped_into_characters():
    tokens = tokenize("我們學習詞彙，也學習國家的歷史。")
    assert {"學習", "詞彙", "國家", "歷史"} <= set(tokens)
    assert "詞" not in tokens and "彙" not in tokens


def test_traditional_text_keeps_its_script():
    assert tokenize("詞彙") == ["詞彙"]  # not converted to 词汇


def test_vocabulary_from_traditional_text_has_levels_and_definitions():
    words = {w.word: w for w in extract_words([Segment(0, 3, "我們學習詞彙。")])}
    assert words["詞彙"].hsk_level == 6 and words["詞彙"].definition.startswith("vocabulary")
    assert words["學習"].hsk_level == 1 and words["學習"].definition == "to learn; to study"


def test_traditional_function_words_are_skipped_like_simplified_ones():
    simplified = {w.word for w in extract_words([Segment(0, 1, "从北京来")])}
    traditional = {w.word for w in extract_words([Segment(0, 1, "從北京來")])}
    assert "从" not in simplified and "從" not in traditional  # 从/從 = "from"
    assert "北京" in simplified and "北京" in traditional
