"""
Experiment 010: sentence-aligned Prakrit-Hindi narrative pairs from the two
prose stories in `word फाइल.docx` (Vajjalagga, stories 'अमंगलिक पुरुष की कथा'
and 'घर में शूर').

Paragraph counts differ between the Prakrit and Hindi sides, so the alignment
is sentence-level: split both sides on danda/question marks, then run a
length-based dynamic program (Gale & Church style, allowing 1-1, 1-2, 2-1, 2-2).

Output: data/prakrit_hindi_narrative_aligned.tsv (all aligned pairs, with
the alignment cost, so low-confidence pairs can be filtered later).

Usage: python experiments/010_narrative_test_set/align_story_pairs.py "word फाइल.docx"
"""
import csv
import html
import math
import re
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "scripts"))
from devlys_to_unicode import devlys_to_unicode  # noqa: E402

OUT = Path("data/prakrit_hindi_narrative_aligned.tsv")
TEST_OUT = Path("data/prakrit_hindi_narrative_test.tsv")
# Blocks dropped after reading the alignment by hand (indices into the DP output).
# 33: pairs a goldsmith-walking line with a wife-returns-home line (wrong).
# 48-55: from here to the end of the second story the alignment is shifted by
#        about one block, so Prakrit sentences are paired with the next Hindi one.
EXCLUDE = {33} | set(range(48, 56))
SENT_SPLIT = re.compile(r"(?<=[।?!])\s*")
NUM_PREFIX = re.compile(r"^\s*\d+\s*\.\s*")
# character-length ratio Hindi/Prakrit measured from the text (prior for the DP)
RATIO = None
SKIP_COST = 6.0
MAX_PAIRS = 2  # 1-1, 1-2, 2-1, 2-2 (sizes 1 or 2 on each side)


def read_paragraphs(docx_path):
    xml = zipfile.ZipFile(docx_path).read("word/document.xml").decode("utf8")
    raw = [html.unescape(re.sub(r"<[^>]+>", "", p))
           for p in re.findall(r"<w:p[ >].*?</w:p>", xml, re.S)]
    return [devlys_to_unicode(p.strip()) for p in raw if p.strip()]


def split_sentences(text):
    text = NUM_PREFIX.sub("", text).strip()
    return [s.strip() for s in SENT_SPLIT.split(text) if s and s.strip()]


def story_sides(paras):
    """Return (prakrit_paragraphs, hindi_paragraphs) for the two prose stories."""
    def find(prefix, start=0):
        for i in range(start, len(paras)):
            if paras[i].startswith(prefix):
                return i
        raise ValueError(prefix)
    # story 1: Prakrit numbered paragraphs follow the Prakrit heading, Hindi
    # numbered paragraphs follow the Hindi heading
    p1 = find("अमंगलियपुरिसस्स कहा")
    h1 = find("अमांगलिक पुरुष की कथा", p1)
    p2 = find("गेहे सूरो")
    h2 = find("घर में शूर", p2)
    prak1 = paras[p1 + 1:h1]
    hind1 = paras[h1 + 1:p2]
    prak2 = paras[p2 + 1:h2]
    hind2 = paras[h2 + 1:]
    return prak1 + prak2, hind1 + hind2


def build_sentences(paras):
    out = []
    for p in paras:
        out.extend(split_sentences(p))
    return out


def ratio_prior(pairs):
    """Median Hindi/Prakrit length ratio over the sentence pairs (for the DP)."""
    rs = sorted(len(h) / max(len(p), 1) for p, h in pairs)
    return rs[len(rs) // 2]


def cost(src_len, tgt_len, ratio):
    if src_len == 0 and tgt_len == 0:
        return 0.0
    if src_len == 0 or tgt_len == 0:
        return SKIP_COST
    return abs(math.log(tgt_len / (src_len * ratio + 1e-9)))


def align(src, tgt, ratio):
    """DP over sentence indices. Blocks of 1 or 2 sentences on each side."""
    n, m = len(src), len(tgt)
    INF = float("inf")
    best = [[INF] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    best[0][0] = 0.0
    moves = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for i in range(n + 1):
        for j in range(m + 1):
            if best[i][j] == INF:
                continue
            for di, dj in moves:
                ni, nj = i + di, j + dj
                if ni > n or nj > m:
                    continue
                s = " ".join(src[i:ni])
                t = " ".join(tgt[j:nj])
                c = best[i][j] + cost(len(s), len(t), ratio) + (0.3 if (di, dj) != (1, 1) else 0.0)
                if c < best[ni][nj]:
                    best[ni][nj] = c
                    back[ni][nj] = (i, j, di, dj)
    pairs = []
    i, j = n, m
    while (i, j) != (0, 0):
        pi, pj, di, dj = back[i][j]
        pairs.append((" ".join(src[pi:i]), " ".join(tgt[pj:j]), best[i][j] - best[pi][pj]))
        i, j = pi, pj
    return list(reversed(pairs))


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    paras = read_paragraphs(sys.argv[1])
    prak_paras, hind_paras = story_sides(paras)
    src = build_sentences(prak_paras)
    tgt = build_sentences(hind_paras)
    print(f"prakrit sentences={len(src)} hindi sentences={len(tgt)}", file=sys.stderr)

    # first pass with a crude ratio prior, then re-estimate from the aligned result
    ratio = 1.0
    for _ in range(2):
        aligned = align(src, tgt, ratio)
        ratio = ratio_prior([(p, h) for p, h, _ in aligned if p and h])
    print(f"length ratio (Hindi/Prakrit) = {ratio:.2f}", file=sys.stderr)

    rows = [(p, h, round(c, 3)) for p, h, c in aligned if p and h]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["prakrit", "hindi", "cost"])
        w.writerows(rows)
    print(f"wrote {len(rows)} aligned blocks -> {OUT}", file=sys.stderr)
    kept = [(p, h) for k, (p, h, _) in enumerate(aligned) if p and h and k not in EXCLUDE]
    with open(TEST_OUT, "w", encoding="utf8", newline="") as f:
        w = csv.writer(f, delimiter="	")
        w.writerow(["prakrit", "hindi"])
        w.writerows(kept)
    print(f"wrote {len(kept)} hand-checked blocks -> {TEST_OUT}", file=sys.stderr)
    skipped = [(p, h) for p, h, _ in aligned if not (p and h)]
    print(f"unaligned blocks (one side empty): {len(skipped)}", file=sys.stderr)


if __name__ == "__main__":
    main()
