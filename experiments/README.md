# Experiment Log — Prakrit MT

Convention: each experiment gets its own numbered folder here with a `report.md`, the exact scripts used, and eval output. Any resulting fine-tuned model is saved as a versioned copy under `models/checkpoints/<version_name>/` (never overwritten in place) so past results stay reproducible.

## Results at a glance

| # | Direction | Zero-shot BLEU / chrF++ | Fine-tuned BLEU / chrF++ |
|---|---|---|---|
| 001 | Prakrit→Hindi | 0.45 / 8.31 | 0.56 / 8.64 |
| 002 | Prakrit→Hindi | 0.62 / 10.87 | 3.73 / 17.30 |
| 003 | Prakrit→Hindi (real gold test) | 0.63 / 12.12 | 1.39 / 19.11 |
| 004 | Prakrit→English | 2.14 / 13.76 | **9.81 / 25.30** ← best English-direction result |
| 005 | Prakrit→Hindi (grammar drills, held out) | 1.03 / 12.20 | **5.06 / 25.15** ← best Hindi-direction result |

## Experiments

**[001 — prakrit_hindi_lora_baseline](001_prakrit_hindi_lora_baseline/report.md)**
First LoRA fine-tune, no Kaggle data yet — synthetic Hindi targets generated via `opus-mt-en-hi` over the existing Prakrit-English pairs. Found and fixed a target-tag-leakage bug mid-run (model was learning to predict literal tag tokens).
Checkpoint: `models/checkpoints/v1_synthetic_hindi_baseline/`

**[002 — repetition_penalty_decoding](002_repetition_penalty_decoding/report.md)**
Decoding-only fix on top of 001's model (no retrain): `repetition_penalty=1.3` + `no_repeat_ngram_size=3` to kill beam-search repetition loops. Revealed real signal that the loops had been masking.
Checkpoint: same as 001.

**[003 — prakrit_hindi_real_kaggle_data](003_prakrit_hindi_real_kaggle_data/report.md)**
First real (non-synthetic) Hindi data: 13 clean pairs from Kaggle's `parallel_corpus` + 478 pairs parsed out of raw Acharanga Bhashyam commentary text. Improves on real gold Hindi, but this data is Jain-philosophy domain — doesn't generalize to the Sattasai-poetry domain used in 001/002 (only 0.76/11.64 there, vs 002's 3.73/17.30).
Checkpoint: `models/checkpoints/v3_real_kaggle_agama_data/`

**[004 — prakrit_english_lora](004_prakrit_english_lora/report.md)**
Switched to the natively-supported Prakrit→English direction (`indictrans2-indic-en-1B`, real `eng_Latn` target, no routing hack needed) using 1,334 real pairs (existing Sattasai + Kaggle `parallel_corpus` + Saman Suttam). **Strongest result in the project**, both quantitatively (4.6x BLEU lift) and qualitatively — coherent output in the correct poetic register vs. zero-shot's garbled transliterations.
Checkpoint: `models/checkpoints/v4_prakrit_english_real_data/`

**[005 — grammar_drills_and_domain_mix](005_grammar_drills_and_domain_mix/report.md)**
Decoded `data/fwd/*.pdf` (10 scanned Hindi↔Prakrit grammar-drill answer sheets, in a legacy DevLys-010 font) into 267 clean real sentence pairs — required reverse-engineering both the font encoding and a `pdftotext`-column-reconstruction row-shift artifact. Combined with Experiment 003's real Agama data into a 3-domain training mix. Result: improves over zero-shot on *all three* test domains (unlike Exp 003's single-domain model), and sets the best Hindi-direction BLEU yet on the grammar-drill domain, with several genuinely correct translations.
Checkpoint: `models/checkpoints/v5_domain_mix_real_data/`

**[006 — vajjalagga_gatha_pairs](006_vajjalagga_gatha_pairs/report.md)**
Data-only: decoded `word फाइल.docx` (a Vajjalagga Prakrit gatha anthology with Hindi `अर्थ` glosses, in the same legacy font as Exp 005's PDFs) into 20 clean Prakrit→Hindi verse pairs (`data/prakrit_hindi_vajjalagga.tsv`). The prose stories in the same document are not extracted, since their paragraphs need sentence alignment. No model trained yet.

**[007 — grammar_drill_only](007_grammar_drill_only/report.md)**
Ablation of Exp 005: fine-tune on the 247 grammar-drill pairs alone (same 20 held-out drills). Drill-only scored 3.71 BLEU / 20.37 chrF++ on the drill test, below the domain mix's 5.06 / 25.15, and it did not improve the other domains. The domain mix stays the best Hindi-direction model. Checkpoint: `models/checkpoints/v6_grammar_drill_only/`.

**[008 — drill_kfold_cv](008_drill_kfold_cv/report.md)**
5-fold cross-validation over all 267 grammar-drill pairs, comparing drill-only and domain-mix training. Pooled over every drill pair: drill-only 4.57 BLEU / 20.96 chrF++, domain mix 1.37 / 17.74. This reverses the 20-example result from Exp 005/007. A drill-validated mix (`mixdv`), run to test a checkpoint-selection confound, finished all five folds: 1.22 BLEU / 16.22 chrF++, so the confound is ruled out and drill-only stays best.

**[009 — phrase_smt_baseline](009_phrase_smt_baseline/report.md)**
Pure-Python phrase-based SMT (IBM Model 1 alignment, phrase extraction, lexical weighting, bigram LM, monotone beam decoder, per-fold weight tuning on a dev split) as a non-neural baseline, on the same 5 drill folds as Exp 008. Tuned without copy-through it scores 4.43 BLEU / 16.12 chrF++ (the untuned baseline was 4.42 / 16.28), close to the drill-only LoRA model (4.57 / 20.96) on BLEU and behind on chrF++. With copy-through it reaches 6.88 BLEU, but 28% of outputs are source copies, so that number is not a fair translation score.

**[010 — narrative_test_set](010_narrative_test_set/report.md)**
47 hand-checked Prakrit-Hindi sentence pairs from the two prose stories in the Vajjalagga document, aligned with a length-based DP and with 9 misaligned blocks dropped (56 automatic, 47 kept). Used as a narrative test set. No model generalizes to it: zero-shot 0.75 BLEU / 13.69 chrF++, domain mix 0.66 / 14.92, drill-only 0.69 / 12.73.
