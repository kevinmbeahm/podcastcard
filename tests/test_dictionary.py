"""Dictionary lookups: CC-CEDICT words + Unihan characters, Traditional input, cross-references."""

import pytest

from src.dictionary import get_definition, lookup, normalize, tone_pinyin
from src.hsk import get_hsk_level


# ---------------------------------------------------------------- pinyin


@pytest.mark.parametrize(
    "numeric, expected",
    [
        ("ni3 hao3", "nǐ hǎo"),
        ("hui4", "huì"),  # the mark goes on the last vowel...
        ("liu2", "liú"),
        ("xiong1", "xiōng"),  # ...but on a or e when present
        ("lu:e4", "lüè"),
        ("nu:3", "nǚ"),
        ("zhong1 guo2", "zhōng guó"),
        ("Xin1 jiang1", "Xīn jiāng"),  # proper nouns keep their capital
        ("ge5", "ge"),  # neutral tone: no mark
        ("r5", "r"),
    ],
)
def test_tone_marks(numeric, expected):
    assert tone_pinyin(numeric) == expected


# ---------------------------------------------------------------- Traditional input


def test_traditional_text_normalises_to_simplified():
    assert normalize("詞彙") == "词汇"
    assert normalize("學習") == "学习"
    assert normalize("頭髮") == "头发"  # a whole-word match fixes what per-character can't
    assert normalize("词汇") == "词汇"  # already simplified
    assert normalize("後來") == "后来"
    assert normalize("hello") == "hello"


@pytest.mark.parametrize("traditional, simplified", [("詞彙", "词汇"), ("學習", "学习"), ("國家", "国家"), ("電腦", "电脑")])
def test_traditional_words_get_the_same_definition_and_hsk_level(traditional, simplified):
    assert get_definition(traditional) == get_definition(simplified) != ""
    assert get_hsk_level(traditional) == get_hsk_level(simplified) > 0


# ---------------------------------------------------------------- cross-references


def test_erhua_and_variant_entries_follow_their_target():
    assert get_definition("小孩儿") == "erhua variant of 小孩: child"
    assert get_definition("好玩儿").endswith("amusing; fun; interesting")
    assert get_definition("干吗").startswith("variant of 干嘛: what are you doing?")
    assert get_definition("㐅") == "old variant of 五: five; 5"


def test_senses_never_leak_raw_cedict_markup():
    for word in ("裡面", "一块儿", "个", "词汇"):
        gloss = get_definition(word)
        assert "[" not in gloss and "|" not in gloss, gloss
    assert get_definition("裡面") == "inside; interior; also pr. (lǐ mian)"


def test_no_hsk_word_is_left_without_a_definition():
    from src.hsk import HSK_WORDS

    assert [w for w in HSK_WORDS if not get_definition(w)] == []


# ---------------------------------------------------------------- characters and phrases


def test_rare_characters_come_from_unihan():
    assert get_definition("㐀") == "(same as 丘) hillock or mound"  # not in CC-CEDICT
    assert get_definition("丌") == '"pedestal" component in Chinese characters'


def test_phrases_made_of_known_words_get_a_literal_gloss():
    assert get_definition("打篮球") == "literally: 打 (to hit) + 篮球 (basketball)"
    assert get_definition("这时候") == "literally: 这 (this) + 时候 (time)"  # not 这时 + 候
    assert get_definition("车上") == "literally: 车 (car) + 上 (up)"


def test_nothing_is_invented_for_unknown_text():
    assert get_definition("xyzzy") == ""
    assert get_definition("") == ""


def test_short_glosses_stay_short_but_full_entries_do_not():
    assert len(get_definition("的")) <= 160
    senses = [s for r in lookup("的")["readings"] for s in r["senses"]]
    assert len(senses) > 3  # the full entry has everything


# ---------------------------------------------------------------- full entries


def test_heteronyms_list_every_reading_with_all_senses():
    readings = {r["pinyin"]: r["senses"] for r in lookup("行")["readings"]}
    assert set(readings) >= {"xíng", "háng"}
    assert any("row; line" in s for s in readings["háng"])
    assert any("to walk" in s for s in readings["xíng"])


def test_full_entry_for_a_traditional_word():
    entry = lookup("詞彙")
    assert entry["word"] == "詞彙" and entry["simplified"] == "词汇"
    assert entry["definition"].startswith("vocabulary")
    assert [c["char"] for c in entry["characters"]] == ["詞", "彙"]
    assert entry["characters"][0]["pinyin"] == "cí"
    # real meanings come before the "variant of …" pointer
    senses = entry["readings"][0]["senses"]
    assert senses[0] == "vocabulary" and senses[-1].startswith("variant of")


def test_traditional_form_is_that_of_the_main_entry():
    assert lookup("词汇")["traditional"] == "詞彙"  # not the variant spelling 詞匯
    assert lookup("播客")["traditional"] is None  # same in both scripts


def test_phrase_without_an_entry_lists_its_components():
    entry = lookup("打篮球")
    assert entry["source"] == "components" and entry["readings"] == []
    assert [(c["text"], c["definition"]) for c in entry["components"]][1] == ("篮球", "basketball")
    assert [c["char"] for c in entry["characters"]] == ["打", "篮", "球"]


def test_single_character_missing_from_cedict_uses_unihan():
    entry = lookup("㐀")
    assert entry["source"] == "unihan"
    assert entry["readings"] == [{"pinyin": "qiū", "senses": ["(same as 丘) hillock or mound"]}]
    assert entry["characters"] == []  # no breakdown for a single character


def test_lookup_of_nothing_is_empty_not_an_error():
    entry = lookup("xyzzy")
    assert entry["definition"] == "" and entry["readings"] == [] and entry["source"] is None


def test_parts_lists_skip_long_explanatory_senses():
    parts = {c["char"]: c["definition"] for c in lookup("打篮球")["characters"]}
    assert parts["打"].startswith("to hit")  # not the grammar note about 打 being "semantically light"
    assert all(len(d) <= 100 for d in parts.values())


def test_a_long_explanatory_sense_does_not_crowd_out_the_real_meanings():
    gloss = get_definition("打")
    assert gloss.startswith("to hit") and "semantically light" not in gloss
    full = [s for r in lookup("打")["readings"] for s in r["senses"]]
    assert any("semantically light" in s for s in full)  # nothing is lost from the full entry
