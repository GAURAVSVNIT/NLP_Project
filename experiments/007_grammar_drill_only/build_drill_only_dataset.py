"""
Build the grammar-drill-only training set for Experiment 007.

Reproduces Exp 005's drill split exactly (same seed, same shuffle call as
build_domain_mix_dataset.py) so the 20-example drill test set is held out
from training and matches the test file already used for Exp 005.
"""
import csv
import random
import sys
from pathlib import Path

DATA_DIR = Path("data")
N_DRILL_TEST = 20


def load_drills():
    with open(DATA_DIR / "prakrit_hindi_grammar_drills.tsv", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        # Exp 005 read these with prakrit/hindi columns; keep the same order.
        return [(r["prakrit"], r["hindi"]) for r in reader]


def main():
    random.seed(123)
    drills = load_drills()
    random.shuffle(drills)
    drill_test = drills[:N_DRILL_TEST]
    drill_train = drills[N_DRILL_TEST:]

    existing = DATA_DIR / "prakrit_hindi_drill_test.tsv"
    with open(existing, encoding="utf-8") as f:
        existing_rows = [(r["prakrit"], r["hindi"]) for r in csv.DictReader(f, delimiter="\t")]
    if existing_rows != drill_test:
        sys.exit("split does not reproduce data/prakrit_hindi_drill_test.tsv; refusing to continue")
    print(f"split reproduces existing drill test ({len(drill_test)} rows)", file=sys.stderr)

    out = DATA_DIR / "prakrit_hindi_drill_only_train.tsv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["prakrit", "hindi"])
        w.writerows(drill_train)
    print(f"wrote {len(drill_train)} training rows -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
