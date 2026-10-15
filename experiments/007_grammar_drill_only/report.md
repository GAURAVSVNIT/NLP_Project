# Experiment 007: Grammar-drill-only fine-tune

**Date:** 2026-10-09
**Checkpoint:** `models/checkpoints/v6_grammar_drill_only/`
**Question:** Does training on the grammar drills alone beat Exp 005's domain mix on the drill domain?

## Setup

- Same hyperparameters, LoRA config and validation split as Exp 005 (r=16, alpha=32, 8 epochs, 40 validation rows).
- Training data: 247 grammar-drill pairs (267 minus the 20 held-out test pairs). The 20 are reproduced exactly by `build_drill_only_dataset.py` (seed 123) and checked against the existing `data/prakrit_hindi_drill_test.tsv`.
- Changes from Exp 005's script: output path, the training data file, and `device_map="cuda"` on model load. The last one was needed because the default CPU-side load failed with a Windows paging-file error (os error 1455).
- Evaluation adds the 20 held-out Vajjalagga pairs (Exp 006) as an out-of-domain check.

## Results (BLEU / chrF++)

| Test set | Zero-shot | Exp 005 domain mix | Exp 007 drill only |
|---|---|---|---|
| Grammar drills (held-out, 20) | 1.03 / 12.20 | **5.06 / 25.15** | 3.71 / 20.37 |
| Agama gold (4) | 0.63 / 12.12 | 0.83 / 17.12 | 0.28 / 9.66 |
| Sattasai synthetic (100) | 0.59 / 10.79 | 1.05 / 11.47 | 0.63 / 9.89 |
| Vajjalagga held-out (20) | 0.35 / 9.49 | not run | 0.24 / 9.64 |

Exp 005 numbers are from `experiments/005_grammar_drills_and_domain_mix/report.md`. The Exp 007 numbers are from `eval_log.txt` and the `eval_results_*.txt` files here.

## Interpretation

- **The domain mix won on the drill domain as well.** Drill-only scored lower than the mix on the drill test (3.71 vs 5.06 BLEU, 20.37 vs 25.15 chrF++). The other two domains gained from the mix, and drill-only did not improve them.
- The drill-only model improved over zero-shot on the drill test, but it fell below zero-shot on Agama gold and only matched it on Sattasai and Vajjalagga, so it does not generalize past its own domain.
- The drill test has only 20 examples, and Agama gold has only 4. Treat these differences as directional, not conclusive.
- The eval log prints "(domain mix)" for this run. That is a label string in the script, copied from Exp 005. The saved `eval_results_*.txt` files are labeled correctly.

## Next steps (superseded)

These were written before Exp 008. The 5-fold cross-validation over all 267 drill pairs found that drill-only (4.57 BLEU) beats the domain mix (1.37 BLEU) on the drill domain, reversing this section's conclusion. See `experiments/008_drill_kfold_cv/report.md`. The 20-example numbers above are kept as a record.

## File inventory

- `build_drill_only_dataset.py`: Builds the 247-pair drill-only training set; reproduces the Exp 005 split and checks it against the existing 20-example test.
- `eval_log.txt`: Evaluation log, zero-shot and fine-tuned, for the v6 run. Its header prints '(domain mix)' by mistake; the result files are labeled correctly.
- `eval_results_agama_gold.txt`: BLEU/chrF++ for drill-only on Agama gold (0.28 / 9.66).
- `eval_results_grammar_drill.txt`: BLEU/chrF++ for drill-only on the 20-example drill test (3.71 / 20.37).
- `eval_results_sattasai_synthetic.txt`: BLEU/chrF++ for drill-only on Sattasai synthetic (0.63 / 9.89).
- `eval_results_vajjalagga_heldout.txt`: BLEU/chrF++ for drill-only on the 20 Vajjalagga pairs (0.24 / 9.64).
- `finetune_lora_drill_only.py`: LoRA fine-tune on drill pairs only (produces the v6 checkpoint). Same hyperparameters as Exp 005.
- `run_eval_drill_only.py`: Evaluates the v6 model on the four test sets, including the Vajjalagga held-out pairs.
- `train_log.txt`: Training log for the v6 run.

## Artifacts outside this folder

- `data/prakrit_hindi_drill_only_train.tsv`: the 247 training pairs written by `build_drill_only_dataset.py`.
- `models/checkpoints/v6_grammar_drill_only/`: the saved LoRA adapter (v6).
