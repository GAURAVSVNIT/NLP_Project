import csv
import json
import re
import sys
from pathlib import Path

DATA_DIR = Path("data")
ARCHIVE_DIR = DATA_DIR / "archive"


def clean_prakrit(text):
    # strip trailing colophon markers and collapse whitespace/blank lines
    text = re.sub(r"त्ति\s*बेमि\s*।?", "", text)
    text = re.sub(r"॥[^॥]*समत्तो\s*॥", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_existing():
    rows = []
    with open(DATA_DIR / "final_train_clean.tsv", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header
        for row in reader:
            if len(row) >= 2 and row[0].strip() and row[1].strip():
                rows.append((row[0].strip(), row[1].strip(), "sattasai_existing"))
    return rows


def clean_hindi(text):
    # the "hindi" field mixes real translation prose with embedded HTML-ish
    # glossary/explanation markup (<sutra-glossary>, <div class="tip">, ...).
    # Keep only the prose that appears before the first such tag.
    if not text:
        return ""
    match = re.search(r"<", text)
    if match:
        text = text[: match.start()]
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_parallel_corpus():
    eng_rows = []
    hindi_rows = []
    seen_hindi = set()
    with open(ARCHIVE_DIR / "prakritnlp_dataset.json", encoding="utf-8") as f:
        data = json.load(f)
    for item in data["components"]["parallel_corpus"]:
        pra = clean_prakrit(item["prakrit"])
        source_tag = f"kaggle_parallel_{item.get('agam','')}_{item.get('location','')}"

        eng = re.sub(r"\s+", " ", item.get("english", "") or "").strip()
        if pra and eng:
            eng_rows.append((pra, eng, source_tag))

        hindi = clean_hindi(item.get("hindi", ""))
        # drop very short/truncated fragments (no real prose translation) and
        # dedupe identical Hindi text reused verbatim across different sutras
        # (data quality issue in the source: some entries share commentary blocks)
        if pra and len(hindi) >= 30 and not hindi.endswith("...") and hindi not in seen_hindi:
            seen_hindi.add(hindi)
            hindi_rows.append((pra, hindi, source_tag))

    return eng_rows, hindi_rows


def load_saman_suttam():
    rows = []
    with open(ARCHIVE_DIR / "prakritnlp_dataset.json", encoding="utf-8") as f:
        data = json.load(f)
    for item in data["components"]["saman_suttam_verses"]:
        pra = clean_prakrit(item.get("prakrit_devanagari", ""))
        eng = re.sub(r"\s+", " ", item.get("english", "") or "").strip()
        if pra and eng:
            rows.append((pra, eng, "kaggle_saman_suttam"))
    return rows


def dedup(rows):
    seen = set()
    deduped = []
    for pra, tgt, source in rows:
        if pra in seen:
            continue
        seen.add(pra)
        deduped.append((pra, tgt, source))
    return deduped


def write_tsv(path, rows, target_col):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["prakrit", target_col, "source"])
        for pra, tgt, source in rows:
            writer.writerow([pra, tgt, source])


def main():
    existing = load_existing()
    parallel_eng, parallel_hindi = load_parallel_corpus()
    saman = load_saman_suttam()

    print(f"existing (Sattasai, Prakrit-English): {len(existing)}", file=sys.stderr)
    print(f"kaggle parallel_corpus (Prakrit-English): {len(parallel_eng)}", file=sys.stderr)
    print(f"kaggle parallel_corpus (Prakrit-Hindi, real): {len(parallel_hindi)}", file=sys.stderr)
    print(f"kaggle saman_suttam (Prakrit-English): {len(saman)}", file=sys.stderr)

    english_combined = dedup(existing + parallel_eng + saman)
    hindi_real = dedup(parallel_hindi)

    print(f"Prakrit-English total after dedup: {len(english_combined)}", file=sys.stderr)
    print(f"Prakrit-Hindi (real, from Agama commentary) after dedup: {len(hindi_real)}", file=sys.stderr)

    write_tsv(DATA_DIR / "prakrit_english_combined.tsv", english_combined, "english")
    write_tsv(DATA_DIR / "prakrit_hindi_real_agama.tsv", hindi_real, "hindi")

    print("Wrote data/prakrit_english_combined.tsv and data/prakrit_hindi_real_agama.tsv", file=sys.stderr)


if __name__ == "__main__":
    main()
