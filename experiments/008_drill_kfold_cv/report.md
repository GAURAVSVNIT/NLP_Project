# Experiment 008: 5-fold cross-validation, drill-only vs domain mix

**Date:** 2026-10-09 to 2026-10-11
**Scripts:** `run_drill_kfold_cv.py`, `run_all_folds.sh`
**Raw outputs:** `cv_results.jsonl` (every held-out prediction for both configs), `run_log.txt`

**Question:** Exp 005 and 007 scored both models on a 20-example drill test, which was too small to decide. Which config is better on the full set of drill pairs?

## Setup

- All 267 grammar-drill pairs, 5 folds (seed 123). Each pair is held out exactly once.
- `drill`: train on the other four folds of drills (about 211 pairs).
- `mix`: the same drill folds plus Agama clean and noisy data (about 700 pairs), with Exp 003's gold test prakrits removed.
- Same LoRA config, learning rate, epochs and generation settings as Exp 005.
- Each (config, fold) ran as its own process. The first full run crashed (exit 139) after drill fold 1, so the driver was changed to one process per fold. Finished folds are skipped on rerun.
- `mixdv` (added later): same training data as `mix`, but checkpoint selection uses 40 drill validation rows. Its run is incomplete (see below).

## Result (pooled over all 267 drill pairs)

| Config | BLEU | chrF++ |
|---|---|---|
| **drill-only** | **4.57** | **20.96** |
| domain mix | 1.37 | 17.74 |

## Interpretation

- On the full set, drill-only beats the domain mix on the drill domain. This reverses the 20-example result in Exp 005 (mix 5.06 vs drill-only 3.71). That result was noise from a tiny test set.
- **Possible confound to check before trusting the reversal:** both configs use the last 40 training rows as the validation set for early stopping. In `drill`, those rows are drill pairs. In `mix`, the shuffled rows are mostly Agama, so the mix config picks its checkpoint by Agama validation loss, which may not match drill quality. This affects Exp 005 and 007 too. A fair comparison would select the checkpoint on drill validation rows in both configs.
- The drill-only result is the stronger model on the drill domain. It does not generalize past that domain (see Exp 007).

## Drill-validated mix (`mixdv`): complete

All five folds finished (update 2026-10-13). Pooled over all 267 drill pairs:

| Config | BLEU | chrF++ |
|---|---|---|
| **drill-only** | **4.57** | **20.96** |
| mix (Agama validation) | 1.37 | 17.74 |
| mixdv (drill validation) | 1.22 | 16.22 |

Checkpoint selection is not the explanation. Selecting on drill validation rows makes the domain mix no better (1.22 vs 1.37 BLEU). The domain mix is clearly worse than drill-only on the drill domain. Drill-only is the best Hindi-direction model in the cross-validation.

Note: the earlier version of this section, written while only folds 0-1 existed, is kept below for the record.

### Earlier partial result (folds 0-1)

Update 2026-10-12: folds 0 and 1 finished. Fold 2 was stopped by the system during low-memory pressure, so folds 2-4 are not run. On the 108 pairs in folds 0-1:

| Config | BLEU | chrF++ |
|---|---|---|
| drill | 3.47 | 21.62 |
| mix (Agama validation) | 1.29 | 18.58 |
| mixdv (drill validation) | 0.55 | 16.61 |

Selecting the checkpoint on drill rows did not rescue the domain mix; on these folds it is worse. That weakens the checkpoint-selection explanation. The domain mix still looks worse than drill-only on the drill domain. This is a partial result, so confirm on all five folds before treating it as settled.

Original plan (below), kept for the record:

Purpose: test whether the `mix` result is an artifact of checkpoint selection on Agama validation rows.

- Fold 0 started: fit=659, val=40 drill rows, test=54. It stalled at about 30 s/step (normal is about 0.4 s/step) because GPU memory was nearly full (5.9 of 6.1 GB) and system RAM was nearly exhausted, so Windows was paging. The process was stopped by hand to protect the machine. No `mixdv` fold was saved.
- A second, unrelated Python process was using about 2.8 GB of RAM and GPU memory: `experiments/run_real.py --config smoke` (PID 8512). It is not part of this experiment and was left running. Close it before rerunning `mixdv`.
- Resume with: `sh experiments/008_drill_kfold_cv/run_all_folds.sh` (it runs only `mixdv`, folds 0-4, and pools the scores).

Until `mixdv` finishes, the `mix` vs `drill` conclusion above has the selection caveat.

## Next steps

1. Clean the DevLys-damaged references (see Exp 009) and rerun both configs.

## File inventory

- `cv_results.jsonl`: Every held-out prediction for the `drill` and `mix` configs (10 folds, 267 pairs each). Source of the pooled scores.
- `run_all_folds.sh`: Shell driver that runs each fold in its own process, then pools the scores. Currently set to `mixdv` only.
- `run_drill_kfold_cv.py`: 5-fold cross-validation driver. Configs: `drill`, `mix`, `mixdv` (mix with drill validation). Resumable; accepts a `config fold` argument to run one fold per process.
- `run_log.txt`: Log of the main CV run (drill and mix), including the first crash (exit 139) and the per-fold reruns.
- `run_log_mixdv.txt`: Log of the incomplete `mixdv` run. Fold 0 stalled at about 30 s/step under GPU and RAM pressure and was stopped by hand.

## Artifacts outside this folder

- `models/cv_008/{drill,mix}_fold{0-4}/`: the Trainer output directories for each fold (intermediate checkpoints). These are not versioned checkpoints and are not used by any later experiment. They can be deleted once the results are final.
