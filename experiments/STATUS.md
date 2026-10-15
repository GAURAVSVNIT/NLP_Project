# Prakrit MT: consolidated status

Last updated: 2026-10-11. This file is the single entry point. Each experiment's
`report.md` has the detail; the index is `README.md`.

## 1. Where the project stands

- **Best Prakrit→English:** Exp 004, LoRA on IndicTrans2 indic-en-1B: 9.81 BLEU / 25.30 chrF++ on the 100-line test.
- **Best Prakrit→Hindi on the drill domain, by cross-validation:** Exp 008, drill-only LoRA: 4.57 BLEU / 20.96 chrF++, pooled over all 267 drill pairs.
- **Best non-neural baseline:** Exp 009, tuned phrase-based SMT with copy-through off: 4.43 BLEU / 16.12 chrF++ (same folds).
- **Resolved:** the domain mix is worse on the drill domain, and the checkpoint-selection explanation is ruled out. The drill-validated run (`mixdv`, all 5 folds) scores 1.22 BLEU / 16.22 chrF++, against 1.37 / 17.74 for mix and 4.57 / 20.96 for drill-only.

## 2. Results table

BLEU / chrF++. "Copies" = outputs that are exact copies of the source.

| Exp | Direction / test | Result | Notes |
|---|---|---|---|
| 001 | PR→HI, synthetic targets | 0.56 / 8.64 fine-tuned (zero-shot 0.45 / 8.31) | Synthetic Hindi targets from opus-mt-en-hi |
| 002 | PR→HI, decoding only | 3.73 / 17.30 | repetition_penalty=1.3, no_repeat_ngram_size=3 |
| 003 | PR→HI, real gold test | 1.39 / 19.11 | Jain-prose domain does not transfer to Sattasai |
| 004 | PR→EN, 100-line test | 9.81 / 25.30 | Best English result |
| 005 | PR→HI, drill test (20) | 5.06 / 25.15 | Domain mix; 20 examples, too small to decide |
| 006 | Vajjalagga gathas | 20 pairs extracted | Data only, no model |
| 007 | PR→HI, drill test (20) | 3.71 / 20.37 | Drill-only; below domain mix on 20 examples |
| 008 | PR→HI, 5-fold CV (267) | drill 4.57 / 20.96; mix 1.37 / 17.74 | Reverses the 20-example result; mixdv (complete): 1.22 / 16.22 |
| 010 | PR→HI, narrative prose (47) | zero-shot 0.75 / 13.69; mix 0.66 / 14.92; drill 0.69 / 12.73 | No model generalizes to narrative |
| 009 | PR→HI, 5-fold CV (267) | SMT tuned, no copy: 4.43 / 16.12 | Copy-through on: 7.45 / 19.27 with 28% copies |

## 3. How to reproduce

All commands run from the project root with `.venv/Scripts/python.exe -X utf8`.

- **Exp 005 training:** `experiments/005_grammar_drills_and_domain_mix/finetune_lora_domain_mix.py`
- **Exp 007 training:** `experiments/007_grammar_drill_only/build_drill_only_dataset.py`, then `finetune_lora_drill_only.py`
- **Exp 008 cross-validation:** `sh experiments/008_drill_kfold_cv/run_all_folds.sh` (set the config list in the script first). The driver is resumable.
- **Exp 009 SMT:** `python experiments/009_phrase_smt_baseline/run_smt_cv.py [--no-copy] [--untuned]`

GPU use: one training run at a time. A training run takes about 2 minutes per fold on the RTX 4050 when the GPU and RAM are free. If training is 50 to 100 times slower than that, check for other GPU or RAM users before doing anything else.

## 4. Known problems

1. **DevLys damage in the references.** The decoded drill references contain spacing and matra errors, for example "छिपगे ा।" and "दाने ांे". Every system is scored against them, so all BLEU values are depressed, and the ranking may also be affected. This is the highest-value data fix.
2. **Copy-through inflates BLEU** for SMT. Prakrit and Hindi share tokens, so copying earns credit. Report the no-copy row.
3. **Tiny test sets.** The Exp 005/007 drill test has 20 examples and the Agama gold test has 4. Use the cross-validation numbers for decisions.
4. **Checkpoint-selection confound** in the `mix` config (Agama validation rows). Tested by `mixdv`: ruled out.
5. **Logging:** the Exp 007 eval log prints "(domain mix)" by mistake. The result files are labeled correctly.

## 5. Open items, in order

1. Clean the DevLys-damaged references and rerun Exp 008 and 009 on the cleaned set.
2. Paired bootstrap significance tests on the saved per-sentence outputs (LoRA vs SMT vs mix).
3. Align the Vajjalagga prose stories (Exp 006) to get more pairs.
4. Back-translation (the plan in `Iterative-Back-Translation-...docx`), which has no code yet.

## 6. Housekeeping

- `models/cv_008/` holds intermediate fold outputs and can be deleted once Exp 008 is final.
- `experiments/009_phrase_smt_baseline/__pycache__/` and `experiments/008_drill_kfold_cv/__pycache__/` are Python caches.
- `scripts/` duplicates scripts in `experiments/`. Pick one copy as canonical.
- `prakrit_mlm_architectures_1.md` is an identical copy of `prakrit_mlm_architectures.md`.
- A separate `run_real.py --config smoke` process (not from this project) was using GPU memory on 2026-10-11. It has since exited.
