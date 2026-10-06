"""Tests for HSK lookup, CEDICT definitions, and word extraction."""

from src.dictionary import get_definition
from src.extract import Segment, extract_words
from src.hsk import HSK_WORDS, get_hsk_level, get_pinyin


def test_hsk_lists_cover_expected_sizes():
    counts = {lv: sum(1 for v in HSK_WORDS.values() if v == lv) for lv in range(1, 7)}
    assert all(counts[lv] > 0 for lv in range(1, 7))
    assert len(HSK_WORDS) > 7000


def test_hsk_levels_for_known_words():
    assert get_hsk_level("喜欢") == 1
    assert get_hsk_level("认识") == 1
    assert get_hsk_level("你好") == 1
    assert get_hsk_level("说") == 1  # via HSK 3.0 fallback
    assert get_hsk_level("词汇") == 6
    assert get_hsk_level("播客") == 0  # not in any HSK list


def test_pinyin_has_tone_marks():
    assert get_pinyin("你好") == "nǐ hǎo"


def test_definition_prefers_real_sense_over_stubs():
    assert get_definition("新").startswith("new")  # not "abbr. for Xinjiang"
    assert "vocabulary" in get_definition("词汇")  # not "variant of ..."
    assert get_definition("播客") == "podcast (loanword)"
    assert get_definition("不存在的词语") == ""


def test_extract_words_filters_noise_and_keeps_context():
    sentence = "我喜欢听中文播客，学习新的词汇。"
    words = {w.word: w for w in extract_words([Segment(0, 3, sentence)])}
    assert "，" not in words and "。" not in words
    assert "我" not in words and "的" not in words  # function words
    assert words["播客"].contexts == [sentence]
    assert words["播客"].definition
    assert words["喜欢"].hsk_level == 1


def test_extract_words_sorts_known_levels_first_unknown_last():
    result = extract_words([Segment(0, 3, "我喜欢听中文播客，学习新的词汇。")])
    levels = [w.hsk_level for w in result]
    known = [lv for lv in levels if lv]
    assert known == sorted(known)
    assert levels[-1] == 0
