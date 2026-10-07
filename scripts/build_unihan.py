"""Regenerate data/unihan.json.gz from the Unicode Han Database (Unihan).

Usage:
    python scripts/build_unihan.py [directory-with-the-two-json-files]

Source: the ``ucd-full`` npm package (https://github.com/iLib-js/UCD), which republishes the
Unicode Character Database as JSON. Without an argument the package is downloaded from the
npm registry and the two files needed are read straight out of the tarball:

    Unihan_Readings.json   English definitions (kDefinition), Mandarin readings (kMandarin)
    Unihan_Variants.json   Traditional <-> Simplified variants

Output (compact, gzip-compressed JSON):
    chars  {character: [english definition, "mandarin readings"]}
    t2s    {traditional character: simplified character}
    s2t    {simplified character: traditional character}

The data is (c) Unicode, Inc. and used under the Unicode License; see data/NOTICE-unihan.txt.
"""

from __future__ import annotations

import gzip
import io
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

REGISTRY_URL = "https://registry.npmjs.org/ucd-full"
OUT_FILE = Path(__file__).parent.parent / "data" / "unihan.json.gz"
FILES = ("Unihan_Readings.json", "Unihan_Variants.json")


def _load_from_directory(directory: Path) -> dict[str, list[dict]]:
    return {
        name: json.loads((directory / name).read_text(encoding="utf-8"))[name.removesuffix(".json")]
        for name in FILES
    }


def _download() -> tuple[str, dict[str, list[dict]]]:
    with urllib.request.urlopen(REGISTRY_URL, timeout=60) as resp:
        meta = json.load(resp)
    version = meta["dist-tags"]["latest"]
    with urllib.request.urlopen(meta["versions"][version]["dist"]["tarball"], timeout=300) as resp:
        blob = resp.read()
    data = {}
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for name in FILES:
            member = tar.extractfile(f"package/{name}")
            data[name] = json.load(member)[name.removesuffix(".json")]
    return version, data


def _char(codepoint: str) -> str:
    """'U+4E22' or 'U+4E22<kFanti' -> the character."""
    return chr(int(codepoint.split("<")[0][2:], 16))


def build(data: dict[str, list[dict]]) -> dict:
    chars: dict[str, list[str]] = {}
    for entry in data["Unihan_Readings.json"]:
        definition = entry.get("kDefinition", "")
        mandarin = entry.get("kMandarin", "")
        if definition or mandarin:
            chars[_char(entry["codepoint"])] = [definition, mandarin]

    t2s: dict[str, str] = {}
    s2t: dict[str, str] = {}
    for entry in data["Unihan_Variants.json"]:
        char = _char(entry["codepoint"])
        for field, target in (("kSimplifiedVariant", t2s), ("kTraditionalVariant", s2t)):
            variants = [_char(v) for v in entry.get(field, "").split()]
            variants = [v for v in variants if v != char]
            if variants:
                target[char] = variants[0]
    return {"chars": chars, "t2s": t2s, "s2t": s2t}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        source, raw = f"directory {sys.argv[1]}", _load_from_directory(Path(sys.argv[1]))
    else:
        version, raw = _download()
        source = f"ucd-full {version}"
    built = build(raw)
    payload = json.dumps(built, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    OUT_FILE.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))
    print(
        f"wrote {OUT_FILE} from {source}: {len(built['chars']):,} characters "
        f"({sum(1 for v in built['chars'].values() if v[0]):,} with a definition), "
        f"{len(built['t2s']):,} traditional->simplified, {len(built['s2t']):,} simplified->traditional; "
        f"{OUT_FILE.stat().st_size / 1e6:.2f} MB"
    )
