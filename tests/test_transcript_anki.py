"""Transcript formatting/annotation and Anki export tests."""

import json
import sqlite3
import zipfile

from src.anki import write_apkg
from src.extract import Segment
from src.transcript import _entry, annotate, format_text, format_vtt, has_cjk, tokenize

SEGS = [
    Segment(0.0, 3.5, "我喜欢听中文播客，学习新的词汇。"),
    Segment(3661.25, 3665.0, "Hello 世界！"),
]


def test_tokens_rejoin_to_original_text():
    for seg in SEGS:
        assert "".join(tokenize(seg.text)) == seg.text


def test_annotate_builds_lexicon_for_every_chinese_token():
    segments, lexicon = annotate(SEGS)
    assert [s["text"] for s in segments] == [s.text for s in SEGS]
    assert "，" not in lexicon and "Hello" not in lexicon  # no punctuation / latin
    assert all(has_cjk(w) for w in lexicon)
    # Function words are NOT filtered here (unlike vocabulary extraction)
    assert "的" in lexicon and lexicon["的"]["definition"]
    assert lexicon["播客"] == {
        "pinyin": "bō kè",
        "hsk_level": 0,
        "definition": "podcast (loanword)",
    }


def test_unknown_compound_gets_per_character_parts():
    entry = _entry("鱼琵汤琶")  # not a real word, so no CEDICT entry
    assert entry["definition"] == ""
    assert [p["char"] for p in entry["parts"]] == list("鱼琵汤琶")
    assert entry["parts"][0]["definition"]  # characters still have glosses
    assert "parts" not in _entry("播客")  # known words don't


def test_vtt_and_text_formats():
    vtt = format_vtt(SEGS).splitlines()
    assert vtt[0] == "WEBVTT"
    assert "00:00:00.000 --> 00:00:03.500" in vtt
    assert "01:01:01.250 --> 01:01:05.000" in vtt
    assert format_text(SEGS).splitlines()[1] == "[61:01] Hello 世界！"


WORDS = [
    {
        "word": "播客",
        "pinyin": "bō kè",
        "definition": "podcast (loanword)",
        "hsk_level": 0,
        "contexts": ["我喜欢听中文播客，学习新的词汇。", "播客<很>好。"],
    },
    {"word": "学习", "pinyin": "xué xí", "definition": "to learn", "hsk_level": 1, "contexts": []},
]


def _notes(apkg):
    with zipfile.ZipFile(apkg) as z:
        db = apkg.parent / "collection.anki2"
        db.write_bytes(z.read("collection.anki2"))
    con = sqlite3.connect(db)
    return con.execute("select flds, tags, guid from notes order by id").fetchall()


def test_apkg_contains_one_note_per_word_with_highlight_and_escaping(tmp_path):
    out = tmp_path / "deck.apkg"
    assert write_apkg(WORDS, out, "PodcastCard::Test", source="Test <ep>") == 2
    notes = _notes(out)
    assert len(notes) == 2
    fields = notes[0][0].split("\x1f")
    assert fields[0] == "播客" and fields[2] == "podcast (loanword)"
    # shortest usable sentence is the main example: highlighted and HTML-escaped
    assert fields[3] == "<b>播客</b>&lt;很&gt;好。"
    assert fields[4] == "我喜欢听中文<b>播客</b>，学习新的词汇。"  # the rest go on the back
    assert fields[6] == "Test &lt;ep&gt;"
    assert "HSK-unlisted" in notes[0][1] and "HSK1" in notes[1][1]


def test_note_guid_is_stable_per_word(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    write_apkg(WORDS, a / "x.apkg", "Deck A")
    write_apkg(WORDS, b / "x.apkg", "Deck B")
    assert [n[2] for n in _notes(a / "x.apkg")] == [n[2] for n in _notes(b / "x.apkg")]
