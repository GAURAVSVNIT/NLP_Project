# Experiment 001: First LoRA fine-tune of IndicTrans2 for Prakrit→Hindi (no Kaggle data yet)

**Date:** 2026-09-06
**Model checkpoint saved at:** `models/checkpoints/v1_synthetic_hindi_baseline/`
**Goal:** Get a first working fine-tuning result without waiting on the PrakritNLP Kaggle dataset, using only data already present in the repo.

## Setup

- Base model: `ai4bharat/indictrans2-indic-indic-1B` (1.2B params), local at `models/indictrans2-indic-indic-1B`
- Environment: `.venv` (Python 3.10), PyTorch 2.11+cu128, `transformers==4.53.2` (pinned — later 4.x/5.x versions break IndicTrans2's `trust_remote_code` custom modeling code), `peft==0.20.0`, `IndicTransToolkit==1.1.1`
- GPU: NVIDIA RTX 4050 Laptop (6GB VRAM)
- Method: script-compatible language routing — Prakrit source text is tagged as `hin_Deva` (per the "Prakrit-Hindi Dataset for IndicTrans" research doc), LoRA adapter (r=16, alpha=32, dropout=0.05, targeting `q_proj`/`v_proj`), base model frozen in bf16.

## Data

No Kaggle/PrakritNLP data used yet. Built a **synthetic** Prakrit→Hindi corpus from existing repo data:

- Source: `data/final_train_clean.tsv` (1,300 Prakrit-English pairs, real human-translated Sattasai-style verses) and `data/100_test_pra.txt` / `100_test_eng.txt` (100-line held-out test set)
- Hindi targets were **machine-generated** from the English side using `Helsinki-NLP/opus-mt-en-hi` (small MarianMT model, ungated, ~300MB) — i.e. Prakrit (real) → English (real, human) → Hindi (synthetic, machine-translated)
- Output: `data/prakrit_hindi_train.tsv` (1,300 rows), `data/prakrit_hindi_test.tsv` (100 rows, held out)
- Train/val split: last 100 of the 1,300 train rows carved out as validation during training; true test set is the separate 100-line file.

**Known limitation:** the Hindi targets are synthetic paraphrases from a weak general-purpose EN→HI model, not gold-standard translations. This caps the ceiling on training signal quality — flagged as the most likely reason results stay poor (see Results).

## Hyperparameters

| Param | Value |
|---|---|
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, v_proj |
| Learning rate | 3e-4 |
| Epochs | 6 |
| Effective batch size | 16 (per-device 4 × grad accum 4) |
| Precision | bf16 |
| Max sequence length | 128 |

## Bug found and fixed mid-experiment

First training run (not saved as a checkpoint) produced a **complete generation collapse** — the fine-tuned model output `"* * * * * * ..."` for every single test example, scoring *worse* than the zero-shot baseline (BLEU 0.00 vs 0.45).

**Root cause:** `IndicProcessor.preprocess_batch()` defaults to `is_target=False`, which prepends the literal tag string `"hin_Deva hin_Deva "` to the text. This is correct for the *source* (encoder input) but I mistakenly also applied it to the **target** (Hindi labels), so the model was trained to predict `"hin_Deva hin_Deva <sentence>"` — teaching it to emit malformed tag tokens as part of every label.

**Fix:** target preprocessing now calls `ip.preprocess_batch(hindi, src_lang=LANG_TAG, is_target=True)` (see `finetune_lora.py` in this folder for the exact code at the time of this run). After the fix, training loss decreased steadily across all 6 epochs (see below) instead of plateauing.

## Results

| | BLEU | chrF++ |
|---|---|---|
| Zero-shot baseline | 0.45 | 8.31 |
| LoRA fine-tuned (this run) | 0.56 | 8.64 |

Eval loss trajectory: 3.95 → 3.67 → 3.55 → 3.52 → 3.49 → 3.49 (steady decrease, no collapse).

**Qualitative assessment (see `eval_results.txt` for full examples):** both models produce grammatically fluent Hindi that is largely **semantically unrelated** to the actual Prakrit meaning.
- Zero-shot: often transliterates Prakrit words into Devanagari nonsense (e.g. `हैलिडापिंजरई`) or fills the max length repeating one phrase.
- Fine-tuned: no longer transliterates garbage, but instead falls into **repetition loops** of a plausible-sounding Hindi phrase (e.g. `अपने पति के साथ, अपने पति के साथ...` repeated 20+ times) — a classic symptom of a model that has absorbed "this should sound like fluent Hindi" but hasn't learned real Prakrit↔Hindi lexical correspondences from too little/too noisy training signal.

## Conclusion

The fine-tuning pipeline (venv → model download → LoRA training → evaluation) works correctly end-to-end on this hardware (6GB VRAM, ~9 min/6 epochs for ~1,200 examples). The tiny BLEU/chrF improvement confirms *some* learning signal is being picked up, but translation quality is not yet usable. This matches the research docs' expectation that ~1,000-1,500 noisy/synthetic pairs are not enough — the real PrakritNLP Kaggle corpus (curated, human-verified Hindi/Sanskrit-Prakrit pairs, larger monolingual data) is likely necessary for meaningfully better results.

## Next steps considered

1. Cheap generation-time fix: add `repetition_penalty` / `no_repeat_ngram_size` to decoding to eliminate the loop artifacts (doesn't fix the lexical gap, but clears noise from the signal) — **see Experiment 002**.
2. Acquire the PrakritNLP Kaggle dataset for real parallel/monolingual data.

## File inventory

- `eval_results.txt`: Zero-shot vs fine-tuned BLEU/chrF++ and sample outputs for this experiment.
- `finetune_lora.py`: LoRA fine-tune of IndicTrans2 indic-indic-1B on the synthetic Prakrit-Hindi pairs (produces v1 checkpoint).
- `make_synthetic_hindi.py`: Builds the synthetic Hindi targets: translates the Prakrit-English pairs' English side into Hindi with `opus-mt-en-hi` (MarianMT).
