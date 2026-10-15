"""
Python port of the DevLys-010 -> Unicode conversion algorithm, transcribed
from the JS source at prakhariitd/COP315-Hindi-fonts-to-Unicode
("devlys 010 (=) Unicode Conversion.html", function convert_to_unicode()).

The substitution tables (devlys_array_one.txt / devlys_array_two.txt) are
extracted programmatically from that JS source (see extract_arrays.py in the
scratchpad) rather than hand-transcribed, to avoid transcription errors in a
227-entry table.

DevLys-010 is a legacy non-Unicode Devanagari font: glyphs are stored in
visual/typing order rather than Unicode's logical order (e.g. the "ि" matra
is typed/stored BEFORE its consonant, and reph "र्" is stored AFTER the
consonant cluster it attaches to). This module reverses that.
"""
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent


def _load_array(name):
    with open(_SCRIPT_DIR / name, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


ARRAY_ONE = _load_array("devlys_array_one.txt")
ARRAY_TWO = _load_array("devlys_array_two.txt")

assert len(ARRAY_ONE) == len(ARRAY_TWO), f"{len(ARRAY_ONE)} vs {len(ARRAY_TWO)}"

SET_OF_MATRAS = set("अ आ इ ई उ ऊ ए ऐ ओ औ ा ि ी ु ू ृ े ै ो ौ ं ः ँ ॅ".split())


def _replace_all(text, array_one, array_two):
    for pat, rep in zip(array_one, array_two):
        if not pat:
            continue
        if pat in text:
            text = text.replace(pat, rep)
    return text


def devlys_to_unicode(text):
    if not text:
        return text

    text = _replace_all(text, ARRAY_ONE, ARRAY_TWO)

    # Glyph1: reph+anusvAr combined glyph
    text = text.replace("±", "Zं")

    # Glyph2: "f" ("ि" chhoti-ee, stored before its consonant) -> moved to
    # after the consonant (Unicode logical order)
    text = text.replace("Æ", "र्f")  # rare variant seen in some texts
    while "f" in text:
        pos = text.index("f")
        if pos + 1 >= len(text):
            text = text[:pos]
            break
        nxt = text[pos + 1]
        text = text[:pos] + nxt + "ि" + text[pos + 2:]

    # Glyph3&4: "fa" ("िं") -> moved two positions forward
    text = text.replace("Ç", "fa")
    text = text.replace("É", "र्fa")
    while "fa" in text:
        pos = text.index("fa")
        if pos + 2 >= len(text):
            text = text[:pos]
            break
        nxt = text[pos + 2]
        text = text[:pos] + nxt + "िं" + text[pos + 3:]

    # Glyph5
    text = text.replace("Ê", "ीZ")

    # Fix "ि्" (wrong chhoti-ee ending up before a half-letter) ->
    # "्" + consonant + "ि"
    while "ि्" in text:
        pos = text.index("ि्")
        if pos + 2 >= len(text):
            break
        consonant = text[pos + 2]
        text = text[:pos] + "्" + consonant + "ि" + text[pos + 3:]

    # Eliminate reph marker "Z": DevLys stores reph ("र्") AFTER the syllable
    # it attaches to; Unicode wants it BEFORE. Find the nearest non-matra
    # run to the left of each "Z" and move "र्" there.
    while True:
        z_pos = text.find("Z")
        if z_pos == -1:
            break
        if z_pos == 0:
            text = text[1:]
            continue
        probable = z_pos - 1
        while probable >= 0 and text[probable] in SET_OF_MATRAS:
            probable -= 1
        if probable < 0:
            probable = 0
        segment = text[probable:z_pos]
        text = text[:probable] + "र्" + segment + text[z_pos + 1:]

    return text


def main():
    import sys

    for line in sys.stdin:
        sys.stdout.write(devlys_to_unicode(line.rstrip("\n")) + "\n")


if __name__ == "__main__":
    main()
