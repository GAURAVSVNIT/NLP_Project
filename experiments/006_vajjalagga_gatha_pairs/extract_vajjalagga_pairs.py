"""
Extract Prakrit -> Hindi verse pairs from the Vajjalagga Word document.

The document ("word फाइल.docx") is a Vajjalagga (Prakrit gatha anthology)
set in a legacy DevLys/Kruti-style Devanagari font. Each gatha is:

    N. <Prakrit line 1>
       <Prakrit line 2> ।। (M)
    अर्थ- <Hindi gloss>

The gloss is a Hindi meaning of the verse, so one gatha = one Prakrit->Hindi pair.

The narrative prose section (the "अमंगलिक पुरुष की कथा" / "घर में शूर" stories)
is NOT extracted: its Hindi paragraphs do not map one-to-one onto its Prakrit
paragraphs, so pairing them needs a sentence aligner, not this parser.

Usage:
    python extract_vajjalagga_pairs.py <path-to-docx> <out.tsv>
"""
import html
import re
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "scripts"))
from devlys_to_unicode import devlys_to_unicode  # noqa: E402

GATHA_END = "।।"
GLOSS_PREFIX = re.compile(r"^अर्थ\s*[-–:]\s*")
NUMBER_PREFIX = re.compile(r"^\s*\d+\s*\.\s*")
GATHA_NUMBER_TAIL = re.compile(r"\s*\(\d+\)\s*$")
STORY_HEADING = re.compile(r"(कहा|कथा)$")  # short heading line that starts the prose stories


def read_paragraphs(docx_path):
    xml = zipfile.ZipFile(docx_path).read("word/document.xml").decode("utf8")
    raw = [html.unescape(re.sub(r"<[^>]+>", "", p))
           for p in re.findall(r"<w:p[ >].*?</w:p>", xml, re.S)]
    return [devlys_to_unicode(p.strip()) for p in raw if p.strip()]


def extract_pairs(paragraphs):
    pairs = []
    verse_lines = []
    stats = {"gathas": 0, "orphan_gloss": 0, "unterminated_verse": 0}
    started = False

    for para in paragraphs:
        if len(para) < 60 and STORY_HEADING.search(para):
            break  # verse section ends where the first prose story starts

        if not started:
            # Skip the document title and anything else before the first numbered gatha.
            if not NUMBER_PREFIX.match(para):
                continue
            started = True

        if GLOSS_PREFIX.match(para):
            gloss = GLOSS_PREFIX.sub("", para).strip()
            if verse_lines:
                pairs.append((" ".join(verse_lines), gloss))
                stats["gathas"] += 1
            else:
                stats["orphan_gloss"] += 1
            verse_lines = []
            continue

        # A numbered line starts a new gatha; discard any unterminated one.
        if NUMBER_PREFIX.match(para):
            if verse_lines:
                stats["unterminated_verse"] += 1
            verse_lines = []
            para = NUMBER_PREFIX.sub("", para)

        verse_lines.append(para)
        if GATHA_END in para:
            # Strip the danda and the trailing "(N)" verse number from the last line;
            # the gloss that follows will pair with the accumulated verse_lines.
            verse_lines[-1] = GATHA_NUMBER_TAIL.sub("", para.split(GATHA_END)[0]).strip()
            verse_lines = [l for l in verse_lines if l]

    return pairs, stats


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    docx_path, out_path = sys.argv[1], sys.argv[2]
    pairs, stats = extract_pairs(read_paragraphs(docx_path))

    with open(out_path, "w", encoding="utf8") as f:
        f.write("prakrit\thindi\n")
        for prakrit, hindi in pairs:
            f.write(f"{prakrit}\t{hindi}\n")

    print(f"pairs written: {len(pairs)} -> {out_path}")
    print(stats)


if __name__ == "__main__":
    main()
