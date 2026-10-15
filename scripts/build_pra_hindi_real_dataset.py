import csv
import random
import sys
from pathlib import Path

DATA_DIR = Path("data")


def load_tsv(path, target_col):
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append((r["prakrit"], r[target_col]))
    return rows


def main():
    random.seed(42)

    clean = load_tsv(DATA_DIR / "prakrit_hindi_real_agama.tsv", "hindi")  # 13, high quality
    noisy = load_tsv(DATA_DIR / "prakrit_hindi_acharanga_bhashyam.tsv", "hindi")  # ~478, noisier OCR-derived

    print(f"clean (parallel_corpus): {len(clean)}", file=sys.stderr)
    print(f"noisy (acharanga_bhashyam parse): {len(noisy)}", file=sys.stderr)

    # dedupe noisy against clean and against itself on prakrit text
    seen = {p for p, _ in clean}
    noisy_deduped = []
    for pra, hi in noisy:
        if pra in seen:
            continue
        seen.add(pra)
        noisy_deduped.append((pra, hi))

    # Held-out test: only from the CLEAN set, since noisy OCR-derived pairs
    # aren't trustworthy enough to serve as ground truth for evaluation.
    random.shuffle(clean)
    n_test = 4
    test_rows = clean[:n_test]
    train_clean = clean[n_test:]

    train_rows = train_clean + noisy_deduped
    random.shuffle(train_rows)

    print(f"train: {len(train_rows)} ({len(train_clean)} clean + {len(noisy_deduped)} noisy)", file=sys.stderr)
    print(f"test (real, clean only): {len(test_rows)}", file=sys.stderr)

    def write(path, rows):
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["prakrit", "hindi"])
            for pra, hi in rows:
                writer.writerow([pra, hi])

    write(DATA_DIR / "prakrit_hindi_real_train.tsv", train_rows)
    write(DATA_DIR / "prakrit_hindi_real_test.tsv", test_rows)
    print("Wrote data/prakrit_hindi_real_train.tsv and data/prakrit_hindi_real_test.tsv", file=sys.stderr)


if __name__ == "__main__":
    main()
