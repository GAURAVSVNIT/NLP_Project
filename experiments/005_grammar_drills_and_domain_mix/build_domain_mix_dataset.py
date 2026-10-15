import csv
import random
import sys
from pathlib import Path

DATA_DIR = Path("data")


def load_tsv(path, hindi_col="hindi", prakrit_col="prakrit"):
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append((r[prakrit_col], r[hindi_col]))
    return rows


def main():
    random.seed(123)

    # grammar drills: Hindi is the source column in this file, so swap order to (prakrit, hindi)
    drills = load_tsv(DATA_DIR / "prakrit_hindi_grammar_drills.tsv", hindi_col="hindi", prakrit_col="prakrit")
    agama_clean = load_tsv(DATA_DIR / "prakrit_hindi_real_agama.tsv")
    agama_noisy = load_tsv(DATA_DIR / "prakrit_hindi_acharanga_bhashyam.tsv")

    print(f"grammar drills: {len(drills)}", file=sys.stderr)
    print(f"agama clean (parallel_corpus): {len(agama_clean)}", file=sys.stderr)
    print(f"agama noisy (acharanga parse): {len(agama_noisy)}", file=sys.stderr)

    random.shuffle(drills)
    n_drill_test = 20
    drill_test = drills[:n_drill_test]
    drill_train = drills[n_drill_test:]

    # reuse the same tiny gold agama test set from experiment 003 for continuity
    agama_test = load_tsv(DATA_DIR / "prakrit_hindi_real_test.tsv")
    agama_test_set = {p for p, _ in agama_test}

    seen = set(agama_test_set)
    train_rows = []
    for pra, hi in drill_train + agama_clean + agama_noisy:
        if pra in seen:
            continue
        seen.add(pra)
        train_rows.append((pra, hi, "mixed"))

    random.shuffle(train_rows)

    print(f"combined train: {len(train_rows)}", file=sys.stderr)
    print(f"drill test (held out, new domain): {len(drill_test)}", file=sys.stderr)
    print(f"agama test (from experiment 003, reused): {len(agama_test)}", file=sys.stderr)

    def write(path, rows, with_source=False):
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            header = ["prakrit", "hindi", "source"] if with_source else ["prakrit", "hindi"]
            writer.writerow(header)
            for row in rows:
                writer.writerow(row if with_source else row[:2])

    write(DATA_DIR / "prakrit_hindi_domain_mix_train.tsv", train_rows, with_source=True)
    write(DATA_DIR / "prakrit_hindi_drill_test.tsv", drill_test)
    print("Wrote data/prakrit_hindi_domain_mix_train.tsv and data/prakrit_hindi_drill_test.tsv", file=sys.stderr)


if __name__ == "__main__":
    main()
