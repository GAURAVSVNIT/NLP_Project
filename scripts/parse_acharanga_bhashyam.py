import csv
import json
import re
import sys
from pathlib import Path

ARCHIVE_DIR = Path("data/archive")
OUT_PATH = Path("data/prakrit_hindi_acharanga_bhashyam.tsv")

SANSKRIT_MARKER = "सं०"
SUTRA_NUM_RE = re.compile(r"([०-९]{1,3})\.\s*")
PAGE_HEADER_NOISE = "आचारांगभाष्यम्"

MAX_PRAKRIT_CHARS = 400
MAX_HINDI_CHARS = 400


def clean(text):
    text = text.replace(PAGE_HEADER_NOISE, " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Devanagari block + common punctuation/whitespace/digits/danda; anything else
# in bulk usually means OCR garbage glyphs leaked in (e.g. stray CJK/symbol chars)
JUNK_CHAR_RE = re.compile(r"[^ऀ-ॿ\sा-ॐ0-9०-९.,;:!?()\[\]'\"\-–—।॥]")


def is_clean_devanagari(text, max_junk=1):
    return len(JUNK_CHAR_RE.findall(text)) <= max_junk


def passes_quality_filters(prakrit, hindi):
    if len(prakrit) < 15 or len(hindi) < 20:
        return False
    if len(hindi) > 8 * len(prakrit):
        return False
    if not is_clean_devanagari(prakrit) or not is_clean_devanagari(hindi):
        return False
    return True


def last_number_split(segment):
    """Find the last numbered-sutra marker in `segment`; return
    (text_before_number, text_after_number) or None if no match."""
    matches = list(SUTRA_NUM_RE.finditer(segment))
    if not matches:
        return None
    last = matches[-1]
    return segment[: last.start()], segment[last.end() :]


def main():
    with open(ARCHIVE_DIR / "acharanga_bhashyam.json", encoding="utf-8") as f:
        data = json.load(f)
    text = data["text"]

    marker_positions = [m.start() for m in re.finditer(re.escape(SANSKRIT_MARKER), text)]
    print(f"Found {len(marker_positions)} '{SANSKRIT_MARKER}' markers", file=sys.stderr)

    # For each marker, find where its Sanskrit chhaya ends (first danda after it)
    sanskrit_ends = []
    for pos in marker_positions:
        after = text[pos + len(SANSKRIT_MARKER) : pos + len(SANSKRIT_MARKER) + 300]
        danda = re.search(r"।", after)
        if danda is None:
            sanskrit_ends.append(None)
        else:
            sanskrit_ends.append(pos + len(SANSKRIT_MARKER) + danda.end())

    rows = []
    cursor = 0
    pending_prakrit = None  # prakrit text for the block whose सं०- we haven't hit yet

    # bootstrap: extract prakrit preceding the very first marker
    if marker_positions and sanskrit_ends[0] is not None:
        first_gap = text[0 : marker_positions[0]]
        split = last_number_split(first_gap[-MAX_PRAKRIT_CHARS:])
        if split:
            pending_prakrit = clean(split[1])

    for i, pos in enumerate(marker_positions):
        s_end = sanskrit_ends[i]
        if s_end is None:
            pending_prakrit = None
            continue

        # Determine the gap between this block's Sanskrit end and the NEXT
        # marker's start (or a fixed window if this is the last marker).
        next_pos = marker_positions[i + 1] if i + 1 < len(marker_positions) else None
        gap_end = next_pos if next_pos is not None else min(len(text), s_end + MAX_HINDI_CHARS + MAX_PRAKRIT_CHARS)
        gap = text[s_end:gap_end]

        split = last_number_split(gap)
        if split:
            hindi_raw, next_prakrit_raw = split
        else:
            hindi_raw, next_prakrit_raw = gap, None

        hindi = clean(hindi_raw[:MAX_HINDI_CHARS])

        if pending_prakrit and passes_quality_filters(pending_prakrit, hindi):
            rows.append((pending_prakrit, hindi))

        if next_prakrit_raw is not None and len(next_prakrit_raw) <= MAX_PRAKRIT_CHARS:
            pending_prakrit = clean(next_prakrit_raw)
        else:
            pending_prakrit = None

    print(f"Extracted {len(rows)} candidate pairs before dedup", file=sys.stderr)

    seen = set()
    deduped = []
    for pra, hi in rows:
        if pra in seen:
            continue
        seen.add(pra)
        deduped.append((pra, hi))

    print(f"After dedup: {len(deduped)}", file=sys.stderr)

    with open(OUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["prakrit", "hindi"])
        for pra, hi in deduped:
            writer.writerow([pra, hi])

    print(f"Wrote {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
