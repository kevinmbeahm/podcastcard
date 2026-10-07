"""Chinese dictionary lookups.

Two bundled sources, used together:

* **CC-CEDICT** (``data/cedict_ts.u8.gz``): ~120k words and phrases with English senses.
* **Unihan** (``data/unihan.json.gz``): English definitions and Mandarin readings for ~44k
  individual characters (including rare ones CC-CEDICT lacks) and the Traditional <->
  Simplified character map.

Lookups accept Simplified or Traditional text. Both files are parsed lazily on first use.
Bump ``DICT_VERSION`` whenever results change, so episodes stored under an older version
refresh their definitions the next time they are opened.
"""

from __future__ import annotations

import gzip
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

from pypinyin import Style, lazy_pinyin

DICT_VERSION = 3  # 3: bare HSK 3.0 characters above level 3 are no longer levelled

_DATA_DIR = Path(__file__).parent.parent / "data"
_CEDICT_FILE = _DATA_DIR / "cedict_ts.u8.gz"
_UNIHAN_FILE = _DATA_DIR / "unihan.json.gz"

# "Traditional Simplified [pin1 yin1] /def one/def two/"
_LINE_RE = re.compile(r"^(\S+) (\S+) \[([^\]]*)\] /(.*)/\s*$")

# Senses that add little on a flashcard: cross-references, proper-noun abbreviations,
# measure-word hints, surnames. They are skipped for the short gloss while real senses exist.
_NOISE_PREFIXES = (
    "CL:",
    "see ",
    "see also",
    "variant of",
    "old variant of",
    "also written",
    "erhua variant",
    "abbr. for",
    "surname",
    "Taiwan pr.",
)

_MAX_DEFINITIONS = 3
_LONG_SENSE = 80  # longer senses are explanations, not glosses; the full entry keeps them
_MAX_COMPONENTS = 4
_MAX_REFERENCE_DEPTH = 3

_CJK = r"㐀-䶿一-鿿豈-﫿\U00020000-\U0003134f"

# "variant of 詞彙|词汇[ci2 hui4]", "see 基友[ji1 you3]", "erhua variant of 一塊|一块[yi1 kuai4]"
_REFERENCE_RE = re.compile(
    r"^(?P<label>[^\[\]|]*?\b(?:variant of|form of|short for|abbr\. for|same as|"
    r"equivalent to|also written|see also|see))\s+"
    rf"(?P<trad>[^\s\[\]|]+)(?:\|(?P<simp>[^\s\[\]]+))?\[(?P<pinyin>[^\]]+)\]"
)
# any "詞彙|词汇[ci2 hui4]" or "位[wei4]" inside a sense, to show it as "词汇 (cí huì)"
_INLINE_REF_RE = re.compile(
    rf"(?P<trad>[{_CJK}]+)(?:\|(?P<simp>[{_CJK}]+))?\[(?P<pinyin>[A-Za-z0-9:,· ]+)\]"
)
# a bare reading such as "also pr. [li3 mian5]"
_BARE_PINYIN_RE = re.compile(r"\[(?P<pinyin>(?:[A-Za-z:]+[1-5] ?)+)\]")
# a measure-word note, with one level of nested parentheses: "(CL:条 (tiáo),尾 (wěi))"
_MEASURE_WORD_RE = re.compile(r"\s*\(CL:[^()]*(?:\([^()]*\)[^()]*)*\)")


class _Entry(NamedTuple):
    traditional: str
    simplified: str
    pinyin: str  # numeric, as written in CC-CEDICT ("ni3 hao3")
    senses: tuple[str, ...]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _cedict() -> tuple[dict[str, list[_Entry]], dict[str, list[_Entry]]]:
    """(entries by simplified headword, entries by traditional headword)."""
    by_simplified: dict[str, list[_Entry]] = {}
    by_traditional: dict[str, list[_Entry]] = {}
    with gzip.open(_CEDICT_FILE, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            m = _LINE_RE.match(line)
            if not m:
                continue
            trad, simp, pinyin, senses = m.groups()
            entry = _Entry(trad, simp, pinyin, tuple(senses.split("/")))
            by_simplified.setdefault(simp, []).append(entry)
            if trad != simp:
                by_traditional.setdefault(trad, []).append(entry)
    return by_simplified, by_traditional


@lru_cache(maxsize=1)
def _unihan() -> dict:
    try:
        with gzip.open(_UNIHAN_FILE, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    except OSError:  # the dictionary still works from CC-CEDICT alone
        return {"chars": {}, "t2s": {}, "s2t": {}}


# ---------------------------------------------------------------------------
# Pinyin and Simplified/Traditional helpers
# ---------------------------------------------------------------------------

_TONE_MARKS = {
    "a": "āáǎàa",
    "e": "ēéěèe",
    "i": "īíǐìi",
    "o": "ōóǒòo",
    "u": "ūúǔùu",
    "ü": "ǖǘǚǜü",
}


def _tone_syllable(syllable: str) -> str:
    """'hui4' -> 'huì', 'lu:e4' -> 'lüè', 'r5' -> 'r'. Anything else is returned unchanged."""
    m = re.fullmatch(r"([A-Za-z:üÜ]+?)([1-5])", syllable)
    if not m:
        return syllable.replace("u:", "ü").replace("U:", "Ü")
    base = m.group(1).replace("u:", "ü").replace("U:", "Ü")
    tone = int(m.group(2))
    if tone == 5:
        return base
    lower = base.lower()
    index = next((lower.index(v) for v in "ae" if v in lower), None)
    if index is None and "ou" in lower:
        index = lower.index("o")
    if index is None:  # otherwise the last vowel carries the mark
        index = next((i for i in range(len(lower) - 1, -1, -1) if lower[i] in "aeiouü"), None)
    if index is None:
        return base
    marked = _TONE_MARKS[lower[index]][tone - 1]
    return base[:index] + (marked.upper() if base[index].isupper() else marked) + base[index + 1 :]


def tone_pinyin(numeric: str) -> str:
    """CC-CEDICT's numeric pinyin ('ni3 hao3') with tone marks ('nǐ hǎo')."""
    return " ".join(_tone_syllable(s) for s in numeric.split())


def _numeric_pinyin(word: str) -> str:
    """pypinyin's reading in CC-CEDICT's style, to pick the right entry of a heteronym."""
    return " ".join(lazy_pinyin(word, style=Style.TONE3, neutral_tone_with_five=True)).lower()


@lru_cache(maxsize=65536)
def normalize(word: str) -> str:
    """Simplified form of *word*, for lookups: 詞彙 -> 词汇, 頭髮 -> 头发.

    Whole words are matched against CC-CEDICT's traditional headwords first (so 頭髮 and
    頭發 both come out right); anything else is converted character by character.
    """
    by_simplified, by_traditional = _cedict()
    if word in by_simplified:
        return word
    if word in by_traditional:
        return by_traditional[word][0].simplified
    t2s = _unihan()["t2s"]
    return "".join(t2s.get(ch, ch) for ch in word)


def simplify_characters(text: str) -> str:
    """Each Traditional character replaced by its Simplified counterpart, one for one.

    Unlike :func:`normalize` this never changes the length of the text, which is what
    segmentation needs; it does not use word-level knowledge (頭髮's 髮 becomes 发 either way).
    """
    t2s = _unihan()["t2s"]
    return "".join(t2s.get(ch, ch) for ch in text)


def _traditional_form(simplified: str) -> str | None:
    """The Traditional spelling of *simplified*, if it differs (from the main entry)."""
    entries = _ranked_entries(simplified)
    if entries:
        main = next((e for e in entries if _clean(e.senses)), entries[0])
        return main.traditional if main.traditional != simplified else None
    s2t = _unihan()["s2t"]
    converted = "".join(s2t.get(ch, ch) for ch in simplified)
    return converted if converted != simplified else None


# ---------------------------------------------------------------------------
# Senses
# ---------------------------------------------------------------------------


def _clean(senses) -> list[str]:
    return [s for s in senses if s and not s.startswith(_NOISE_PREFIXES)]


def _pretty(sense: str) -> str:
    """Show CC-CEDICT's inline references readably: '詞彙|词汇[ci2 hui4]' -> '词汇 (cí huì)'."""
    sense = _INLINE_REF_RE.sub(
        lambda m: f"{m['simp'] or m['trad']} ({tone_pinyin(m['pinyin'])})", sense
    )
    return _BARE_PINYIN_RE.sub(lambda m: f"({tone_pinyin(m['pinyin'].strip())})", sense)


def _ranked_entries(simplified: str) -> list[_Entry]:
    """Every CC-CEDICT entry for the headword, the reading pypinyin expects first."""
    entries = _cedict()[0].get(simplified, [])
    if len(entries) < 2:
        return list(entries)
    wanted = _numeric_pinyin(simplified)
    # CC-CEDICT capitalises proper nouns ('Xin1' vs 'xin1'): prefer the exact reading
    return sorted(entries, key=lambda e: (e.pinyin != wanted, e.pinyin.lower() != wanted))


def _unihan_definition(char: str) -> str:
    chars = _unihan()["chars"]
    record = chars.get(char) or chars.get(normalize(char))
    return record[0] if record else ""


def _resolve_reference(senses, depth: int) -> str:
    """Follow 'variant of X' / 'see X' to X's own definition: 'variant of 词汇: vocabulary; …'."""
    for sense in senses:
        m = _REFERENCE_RE.match(sense)
        if not m:
            continue
        target = m["simp"] or m["trad"]
        gloss = _definition(target, depth + 1)
        if gloss:
            return f"{m['label']} {target}: {gloss}"
    return ""


def _brief(gloss: str) -> str:
    """One short, plain meaning out of a gloss, for the 'literally: …' form.

    The first sense is the main one; only a long first sense (usually a grammar note, as for
    打) is replaced by the shortest of the first three.
    """
    options = []
    for part in gloss.split(";")[:3]:
        part = _MEASURE_WORD_RE.sub("", part).strip()
        part = re.sub(r"^(\([^)]*\)\s*)+", "", part) or part  # drop "(bound form)"
        if part:
            options.append(part)
    if not options:
        return gloss
    pick = options[0] if len(options[0]) <= 28 else min(options, key=len)
    return pick if len(pick) <= 36 else pick[:35].rstrip() + "…"


def _summary(gloss: str, limit: int = 100) -> str:
    """A compact gloss for the parts lists: skips long explanatory senses (as for 打), keeps a few."""
    senses = [part.strip() for part in gloss.split(";") if part.strip()]
    short = [part for part in senses if len(part) <= 50]
    return _shorten("; ".join((short or senses)[:3]), limit)


def _shorten(gloss: str, limit: int = 160) -> str:
    """Keep a short gloss tidy: cut at a sense boundary (the full entry has everything)."""
    if len(gloss) <= limit:
        return gloss
    kept = ""
    for part in gloss.split("; "):
        candidate = f"{kept}; {part}" if kept else part
        if len(candidate) > limit:
            break
        kept = candidate
    return kept or gloss[: limit - 1].rstrip() + "…"


def _components(simplified: str) -> list[tuple[str, str]]:
    """Split a word that is not an entry into dictionary words: [(text, gloss), …].

    Picks the most probable split by word frequency (jieba's), so 这时候 becomes 这 + 时候
    rather than 这时 + 候. Single characters are always allowed as a last resort.
    """
    import math

    import jieba

    jieba.initialize()
    frequency, total = jieba.dt.FREQ, jieba.dt.total or 1
    by_simplified, _ = _cedict()
    n = len(simplified)
    best: list[tuple[float, int]] = [(-math.inf, 0)] * (n + 1)
    best[0] = (0.0, 0)
    for i in range(n):
        if best[i][0] == -math.inf:
            continue
        for j in range(i + 1, min(n, i + 6) + 1):
            piece = simplified[i:j]
            if (i, j) == (0, n) and n > 1:
                continue  # the word itself is what we are explaining
            if j > i + 1 and piece not in by_simplified:
                continue
            score = best[i][0] + math.log((frequency.get(piece) or 1) / total)
            if score > best[j][0]:
                best[j] = (score, i)
    parts: list[tuple[str, str]] = []
    j = n
    while j > 0:
        i = best[j][1]
        parts.append((simplified[i:j], _definition(simplified[i:j], _MAX_REFERENCE_DEPTH - 1, compose=False)))
        j = i
    return parts[::-1]


def _definition(word: str, depth: int = 0, compose: bool = True) -> str:
    simplified = normalize(word)
    entries = _ranked_entries(simplified)

    for entry in entries:  # 1. real senses
        cleaned = _clean(entry.senses)
        if cleaned:
            short = [sense for sense in cleaned if len(sense) <= _LONG_SENSE] or cleaned
            return "; ".join(_pretty(sense) for sense in short[:_MAX_DEFINITIONS])
    if depth < _MAX_REFERENCE_DEPTH:  # 2. "variant of …" -> the target's definition
        for entry in entries:
            resolved = _resolve_reference(entry.senses, depth)
            if resolved:
                return resolved
    if len(simplified) == 1:  # 3. rare characters CC-CEDICT lacks
        gloss = _unihan_definition(simplified)
        if gloss:
            return gloss
    if entries:  # 4. only stubs left (a surname, an abbreviation): better than nothing
        return "; ".join(_pretty(s) for s in entries[0].senses[:_MAX_DEFINITIONS])
    if compose and len(simplified) > 1:  # 5. a phrase made of known words
        parts = _components(simplified)
        if len(parts) <= _MAX_COMPONENTS and all(gloss for _, gloss in parts):
            return "literally: " + " + ".join(f"{text} ({_brief(gloss)})" for text, gloss in parts)
    return ""


@lru_cache(maxsize=100_000)
def get_definition(word: str) -> str:
    """A short English gloss for *word* (Simplified or Traditional), or ``""`` if unknown.

    Preference: CC-CEDICT's senses (the reading pypinyin expects first), then the target of a
    "variant of …" entry, then Unihan for single characters, then a surname/abbreviation
    stub, and finally, for a phrase made of known words, a literal word-by-word gloss.
    """
    return _shorten(_definition(word))


# ---------------------------------------------------------------------------
# Full entries
# ---------------------------------------------------------------------------


def _character_readings(char: str) -> list[str]:
    """Distinct Mandarin readings of one character, with tone marks."""
    simplified = normalize(char)
    readings = [tone_pinyin(e.pinyin) for e in _ranked_entries(simplified)]
    if not readings:
        record = _unihan()["chars"].get(char) or _unihan()["chars"].get(simplified)
        readings = record[1].split() if record and record[1] else []
    return list(dict.fromkeys(r for r in readings if r.lower() != "xx5"))


def lookup(word: str) -> dict:
    """Everything the dictionary knows about *word*, for the full-entry view.

    ``readings``: each pronunciation with all of its senses (cross-references made readable);
    ``components``: for a phrase that is not itself an entry, the words it is made of;
    ``characters``: each character of a multi-character word. ``source`` says where the
    meaning came from: ``"cedict"``, ``"unihan"``, ``"components"`` or ``None``.
    """
    simplified = normalize(word)
    entries = _ranked_entries(simplified)

    readings: list[dict] = []
    for entry in entries:
        pinyin = tone_pinyin(entry.pinyin)
        reading = next((r for r in readings if r["pinyin"] == pinyin), None)
        if reading is None:
            reading = {"pinyin": pinyin, "senses": [], "_stubs": []}
            readings.append(reading)
        for sense in filter(None, entry.senses):
            noise = sense.startswith(_NOISE_PREFIXES)
            reading["_stubs" if noise else "senses"].append(_pretty(sense))
    for reading in readings:  # real meanings first, then "variant of …", surnames, measure words
        reading["senses"] += reading.pop("_stubs")

    source = "cedict" if readings else None
    if not readings and len(simplified) == 1:
        gloss = _unihan_definition(simplified)
        if gloss:
            source = "unihan"
            pinyin = " / ".join(_character_readings(simplified))
            readings = [{"pinyin": pinyin, "senses": [s.strip() for s in gloss.split(";") if s.strip()]}]

    components: list[dict] = []
    if not entries and len(simplified) > 1:
        for text, gloss in _components(simplified):
            components.append(
                {
                    "text": text,
                    "pinyin": " ".join(lazy_pinyin(text, style=Style.TONE)),
                    "definition": _summary(gloss),
                }
            )
        if not readings and get_definition(word):
            source = "components"

    characters: list[dict] = []
    if len(word) > 1:
        for char in dict.fromkeys(word):
            if re.match(rf"[{_CJK}]", char):
                characters.append(
                    {
                        "char": char,
                        "pinyin": " / ".join(_character_readings(char)),
                        "definition": _summary(get_definition(char)),
                    }
                )

    return {
        "word": word,
        "simplified": simplified,
        "traditional": _traditional_form(simplified),
        "definition": get_definition(word),
        "source": source,
        "readings": readings,
        "components": components,
        "characters": characters,
    }
