# Experiment 009: Phrase-based SMT baseline (non-neural)

**Date:** 2026-10-11
**Code:** `phrase_smt.py` (pure Python), `run_smt_cv.py` (driver)
**Outputs:** `smt_cv_results.jsonl` (copy-through on), `smt_cv_results_nocopy.jsonl` (copy-through off)
**Purpose:** a non-neural baseline for the LoRA results (Exp 008), as suggested in the back-translation plan's "use SMT" item.

## What was built

Moses is not installed here, so `phrase_smt.py` reimplements the core pipeline:
1. IBM Model 1 word alignment (10 EM iterations), in both directions
2. Symmetrization: intersection of best links, plus union points that touch an unaligned word
3. Consistent phrase-pair extraction (Koehn et al. 2003), max length 3
4. Relative-frequency phrase scores, p(e|f) and p(f|e), top-5 translations per source phrase
5. Interpolated bigram LM over training targets
6. Monotone beam-search decoder (no reordering), beam 60

Weights are fixed by hand (TM 1.0, LM 0.5, word bonus 0.3, phrase penalty −0.2). There is no MERT tuning, no lexical weighting, and no reordering model.

## Setup

Same 5 folds as Exp 008 (seed 123, 267 drill pairs). Each fold trains on the other ~211 drill pairs only, which matches the `drill` config. Scores are pooled over all 267 held-out pairs.

## Results (BLEU / chrF++, pooled over 267)

| System | BLEU | chrF++ | Outputs that are exact copies of the source |
|---|---|---|---|
| SMT, copy-through on | 6.88 | 19.51 | 76 / 267 (28%) |
| **SMT, copy-through off** | **4.42** | **16.28** | 0 / 267 |
| LoRA drill-only (Exp 008) | 4.57 | 20.96 | 0 / 534 |
| LoRA domain mix (Exp 008) | 1.37 | 17.74 | 0 / 534 |

## Interpretation

- **Copy-through inflates BLEU.** Prakrit and Hindi share many tokens (वह, तुम, हम, and so on). When the decoder can't cover a word, copying it earns BLEU credit without being a translation. That is why the first SMT number (6.88) looks better than the LoRA model. The honest baseline is the no-copy row.
- **Fair comparison:** SMT without copying (4.42 BLEU) is close to the drill-only LoRA model (4.57), but the LoRA model is clearly ahead on chrF++ (20.96 vs 16.28). A real neural model at 267 pairs beats the phrase baseline on character-level overlap, but not by much on BLEU.
- **Reference quality is a shared ceiling.** Many references are damaged by the DevLys decoding, with broken matras and misplaced spaces (for example "छिपगे ा।", "दाने ांे"). Every system is scored against those errors. Cleaning the references would raise all scores, and it would be the first step before drawing finer conclusions.
- **No leakage:** none of the 54 fold-0 test references appears verbatim in fold-0 training.

## Limitations

- Monotone decoding: Prakrit and Hindi word order differs enough that this costs quality.
- Untuned weights: MERT would likely help, but with 211 pairs the tuning set would be tiny.
- Tiny LM: trained on the same ~211 target sentences.

## Next steps

1. Clean the DevLys-damaged references, then rerun Exp 008 and 009 on the cleaned set.
2. Compare the SMT and LoRA systems with paired bootstrap significance tests on the saved per-sentence outputs.

## Update (2026-10-11): lexical weighting and per-fold weight tuning

Code changes:
- `phrase_smt.py`: lexical weight lex(e|f) (Koehn's formula, averaged over phrase occurrences) added as a feature. Weights are passed into the decoder instead of read from globals.
- `run_smt_cv.py`: per fold, 20% of the training pairs are held out as a dev split, feature weights are tuned by coordinate search on dev BLEU (2 rounds over a 5-feature grid), then the model is retrained on all training pairs and decodes the untouched test fold. Chosen weights are saved in `smt_*_weights.json`.

| System (pooled over 267) | BLEU | chrF++ | Copies |
|---|---|---|---|
| Untuned, lex on, no copy | 3.93 | 15.59 | 0 |
| **Tuned, lex on, no copy** | **4.43** | **16.12** | 0 |
| Tuned, lex on, copy-through | 7.45 | 19.27 | 76 / 267 |
| Earlier untuned, no lex, no copy | 4.42 | 16.28 | 0 |

Findings:
- Tuning gives +0.5 BLEU over the same untuned features, within noise. Dev splits have only about 42 sentences, so the tuned weights are noisy.
- The lex feature did not help on this data. The tuner set lex to 0 in four of five folds. The untuned lex run is worse than the earlier no-lex run, which suggests the default lex weight was too large.
- Copy-through still inflates BLEU (7.45 with 28% copies vs 4.43 without). The no-copy row is the honest comparison against the LoRA model (4.57 BLEU / 20.96 chrF++).

Files: `smt_tuned_cv_results.jsonl`, `smt_tuned_nocopy_cv_results.jsonl`, `smt_untuned_nocopy_cv_results.jsonl`, and the matching `*_weights.json` files.

## File inventory

- `phrase_smt.py`: Pure-Python phrase-based SMT: IBM Model 1 alignment, phrase extraction, lexical weighting, bigram LM, monotone beam decoder.
- `run_smt_cv.py`: Driver for the SMT baseline on the Exp 008 folds. Flags: `--no-copy`, `--untuned`. Tunes weights per fold on a dev split.
- `smt_cv_results.jsonl`: First untuned SMT run, copy-through on (BLEU 6.88, 76 copies). Superseded by the tuned runs.
- `smt_cv_results_nocopy.jsonl`: First untuned SMT run, no copy-through (4.42 BLEU). Superseded.
- `smt_tuned_cv_results.jsonl`: Tuned SMT, copy-through on (7.45 BLEU, 76 copies).
- `smt_tuned_nocopy_cv_results.jsonl`: Tuned SMT, no copy-through (4.43 BLEU / 16.12 chrF++). The fair comparison.
- `smt_tuned_nocopy_weights.json`: Per-fold tuned weights and dev BLEU, no copy-through.
- `smt_tuned_weights.json`: Per-fold tuned weights and dev BLEU, copy-through on.
- `smt_untuned_nocopy_cv_results.jsonl`: Untuned SMT with the lex feature, no copy-through (3.93 BLEU).
- `smt_untuned_nocopy_weights.json`: Per-fold weights used by the untuned run (defaults), with dev BLEU empty.

## Artifacts outside this folder

- `experiments/009_phrase_smt_baseline/__pycache__/` is a Python cache. It is safe to delete.
