"""
Experiment 009 driver: phrase-based SMT on the same 5 drill folds as Exp 008
(seed 123), trained on drill pairs only.

Per fold:
  1. Hold out 20% of the training pairs as a dev split (the test fold is never touched).
  2. Train on the 80% part, tune feature weights by coordinate search on dev BLEU.
  3. Retrain on all training pairs with the tuned weights and decode the test fold.

Flags:
  --no-copy   drop unknown source words instead of copying them (honest ablation)
  --untuned   skip tuning and use DEFAULT_WEIGHTS (reproduces the first baseline)

Usage (from project root): python experiments/009_phrase_smt_baseline/run_smt_cv.py [--no-copy] [--untuned]
"""
import csv
import json
import random
import sys
from pathlib import Path

import sacrebleu

sys.path.insert(0, str(Path(__file__).parent))
from phrase_smt import DEFAULT_WEIGHTS, PhraseModel, tokenize  # noqa: E402

DATA = Path("data")
N_FOLDS = 5
DEV_FRAC = 0.2
GRID = {
    "tm": [0.5, 1.0, 1.5],
    "lex": [0.0, 0.5, 1.0],
    "lm": [0.25, 0.5, 1.0],
    "word": [0.0, 0.3, 0.6],
    "phrase": [-0.5, -0.2, 0.0],
}
TUNE_ROUNDS = 2

NO_COPY = "--no-copy" in sys.argv
UNTUNED = "--untuned" in sys.argv
TAG = ("_untuned" if UNTUNED else "_tuned") + ("_nocopy" if NO_COPY else "")
OUT = Path(f"experiments/009_phrase_smt_baseline/smt{TAG}_cv_results.jsonl")
WEIGHTS_OUT = Path(f"experiments/009_phrase_smt_baseline/smt{TAG}_weights.json")


def load_drills():
    with open(DATA / "prakrit_hindi_grammar_drills.tsv", encoding="utf-8") as f:
        return [(r["prakrit"], r["hindi"]) for r in csv.DictReader(f, delimiter="\t")]


def make_folds(drills):
    # identical to run_drill_kfold_cv.make_folds
    rng = random.Random(123)
    idx = list(range(len(drills)))
    rng.shuffle(idx)
    size = (len(idx) + N_FOLDS - 1) // N_FOLDS
    return [[drills[i] for i in idx[k * size:(k + 1) * size]] for k in range(N_FOLDS)]


def decode(model, pairs, weights):
    hyps = [model.translate(tokenize(p), w=weights, copy_oov=not NO_COPY) for p, _ in pairs]
    return hyps


def dev_bleu(model, dev, weights):
    hyps = decode(model, dev, weights)
    return sacrebleu.corpus_bleu(hyps, [[h for _, h in dev]]).score


def tune(train):
    """Coordinate search on dev BLEU; returns the best weight dict."""
    rng = random.Random(7)
    shuffled = list(train)
    rng.shuffle(shuffled)
    n_dev = max(1, int(len(shuffled) * DEV_FRAC))
    dev, fit = shuffled[:n_dev], shuffled[n_dev:]
    model = PhraseModel([(tokenize(p), tokenize(h)) for p, h in fit])

    best = dict(DEFAULT_WEIGHTS)
    best_score = dev_bleu(model, dev, best)
    for _ in range(TUNE_ROUNDS):
        for name, values in GRID.items():
            for v in values:
                cand = dict(best, **{name: v})
                score = dev_bleu(model, dev, cand)
                if score > best_score:
                    best, best_score = cand, score
    return best, best_score


def main():
    drills = load_drills()
    folds = make_folds(drills)
    assert sum(len(f) for f in folds) == len(drills)

    rows_out = []
    chosen = []
    for k, test in enumerate(folds):
        held = {p for p, _ in test}
        train = [(p, h) for p, h in drills if p not in held]

        if UNTUNED:
            weights, dev_score = dict(DEFAULT_WEIGHTS), None
        else:
            weights, dev_score = tune(train)
        chosen.append({"fold": k, "weights": weights, "dev_bleu": dev_score})

        model = PhraseModel([(tokenize(p), tokenize(h)) for p, h in train])
        hyps = decode(model, test, weights)
        print(f"fold {k}: train={len(train)} test={len(test)} dev_bleu={dev_score} weights={weights}", file=sys.stderr)
        rows_out.append({
            "config": "smt" + TAG, "fold": k,
            "prakrit": [p for p, _ in test],
            "ref": [h for _, h in test],
            "hyp": hyps,
        })

    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows_out), encoding="utf-8")
    WEIGHTS_OUT.write_text(json.dumps(chosen, ensure_ascii=False, indent=2), encoding="utf-8")

    hyps = [h for r in rows_out for h in r["hyp"]]
    refs = [x for r in rows_out for x in r["ref"]]
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)
    copies = sum(1 for r in rows_out for p, h in zip(r["prakrit"], r["hyp"]) if tokenize(p) == tokenize(h))
    print(f"POOLED smt{TAG}: n={len(refs)} BLEU={bleu.score:.2f} chrF++={chrf.score:.2f} copies={copies}", file=sys.stderr)


if __name__ == "__main__":
    main()
