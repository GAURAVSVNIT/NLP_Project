"""
Parse the decoded (data/fwd_decoded/*.txt) Hindi<->Prakrit grammar drill
answer sheets into clean sentence pairs.

Structure: each file has one or more numbered sections whose instruction
line contains "अनुवाद" (translate) or "कृदन्त" (participle) followed by
numbered items "<N> <hindi sentence>   <prakrit answer>" (columns separated
by a wide whitespace gap from pdftotext -layout).

Known quirk (verified by hand against several files): some sections have a
one-row COLUMN SHIFT, where the Prakrit answer printed on row i actually
belongs to row (i+1)'s Hindi sentence — caused by pdftotext's column
reconstruction absorbing the first real answer onto the (answer-less)
instruction header row. Detected via whether the header line itself carries
trailing content past its own sentence-ending punctuation. The shift chain
breaks (and resets to direct same-row pairing) whenever an item's own row
has no visible Prakrit answer (a blank most likely caused by a page break).
"""
import csv
import re
import sys
from collections import deque
from pathlib import Path

DECODED_DIR = Path("data/fwd_decoded")
OUT_PATH = Path("data/prakrit_hindi_grammar_drills.tsv")

ITEM_RE = re.compile(r"^\s*(\d+)\s+(.*)$")
SECTION_HEADER_RE = re.compile(r"^\s*(\d+)\.\s*(.*)$")
# section boundaries are sometimes OCR'd with "/" or other punctuation
# instead of "." (e.g. "2/ वाक्य बनाओ।" in file 09) - used only to detect
# where a translation section ENDS, not to decide what to extract
ANY_SECTION_BOUNDARY_RE = re.compile(r"^\s*\d+\s*[.\/@:]\s")
INSTRUCTION_KEYWORDS = ("अनुवाद", "कृदन्त")
NOISE_TITLE_RE = re.compile(r"प्रश्न-पत्र-?\s*\d*")


def split_columns(line):
    parts = [p.strip() for p in re.split(r"\s{2,}", line.strip()) if p.strip()]
    return parts


def clean(text):
    text = NOISE_TITLE_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


ORPHAN_INDENT_THRESHOLD = 20  # leading spaces suggesting content sits in the answer column


def parse_section(lines):
    """lines: the header line followed by all lines until the next section
    header (or end of file). Returns list of (hindi, prakrit) pairs.

    Handles two known quirks (verified by hand against several files):
    1. A one-row column SHIFT (see module docstring), detected via whether
       the header line carries trailing content past its own instruction.
    2. Occasional ORPHAN answer-only rows with no numbered question (large
       leading indent placing them in the answer column) that add an extra
       step to the shift depth (seen in file 06) - these feed an extra
       entry into the pending queue rather than being skipped.
    """
    header_cols = split_columns(lines[0])
    header_trailing = clean(header_cols[1]) if len(header_cols) > 1 else ""

    events = []  # ("item", hindi, prakrit_or_None) | ("orphan", prakrit)
    for line in lines[1:]:
        m = ITEM_RE.match(line)
        if m:
            cols = split_columns(line)
            if not cols:
                continue
            hindi = clean(re.sub(r"^\d+\s*", "", cols[0]))
            prakrit = clean(cols[1]) if len(cols) > 1 else ""
            if hindi:
                events.append(("item", hindi, prakrit or None))
            continue

        stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent >= ORPHAN_INDENT_THRESHOLD:
            events.append(("orphan", clean(stripped)))

    pairs = []
    shift_mode = bool(header_trailing)
    pending = deque([header_trailing] if shift_mode else [])

    for event in events:
        if event[0] == "orphan":
            # an orphan answer-only row can appear anywhere in a section
            # (seen right after the header in file 06, but mid-list in file
            # 08) - wherever it occurs, it means the running row alignment
            # is now off by one extra step from this point on.
            pending.append(event[1])
            shift_mode = True
            continue

        _, hindi, prakrit = event
        if shift_mode:
            if pending:
                pairs.append((hindi, pending.popleft()))
            if prakrit:
                pending.append(prakrit)
            else:
                pending.clear()
                shift_mode = False
        else:
            if prakrit:
                pairs.append((hindi, prakrit))

    return pairs


def find_sections(all_lines):
    """Return list of (start_idx, end_idx) for each instruction section
    whose header contains a translation-instruction keyword."""
    header_positions = []
    for i, line in enumerate(all_lines):
        m = SECTION_HEADER_RE.match(line)
        if m and any(kw in line for kw in INSTRUCTION_KEYWORDS):
            header_positions.append(i)

    sections = []
    for idx, start in enumerate(header_positions):
        # section ends at the next ANY numbered header ("N. ...." / "N/ ...."
        # / etc), not just translation ones, or end of file
        end = len(all_lines)
        for j in range(start + 1, len(all_lines)):
            if ANY_SECTION_BOUNDARY_RE.match(all_lines[j]):
                end = j
                break
        sections.append((start, end))
    return sections


def main():
    all_pairs = []
    for f in sorted(DECODED_DIR.glob("*.txt")):
        text = f.read_text(encoding="utf-8")
        lines = text.split("\n")
        sections = find_sections(lines)
        file_pair_count = 0
        for start, end in sections:
            section_lines = lines[start:end]
            pairs = parse_section(section_lines)
            all_pairs.extend((h, p, f.stem) for h, p in pairs)
            file_pair_count += len(pairs)
        print(f"{f.name}: {len(sections)} section(s), {file_pair_count} pairs", file=sys.stderr)

    # dedupe on (hindi, prakrit) pair
    seen = set()
    deduped = []
    for hindi, prakrit, source in all_pairs:
        key = (hindi, prakrit)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((hindi, prakrit, source))

    print(f"\nTotal pairs: {len(all_pairs)}, after dedup: {len(deduped)}", file=sys.stderr)

    with open(OUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["hindi", "prakrit", "source"])
        for hindi, prakrit, source in deduped:
            writer.writerow([hindi, prakrit, source])
    print(f"Wrote {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
