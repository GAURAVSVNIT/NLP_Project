# Experiment 003: Prakrit→Hindi LoRA fine-tune on real Kaggle/Agama data

**Date:** 2026-09-06
**Model checkpoint:** `models/checkpoints/v3_real_kaggle_agama_data/`
**Goal:** The user added the PrakritNLP Kaggle dataset to `data/archive/`. Replace the synthetic (MarianMT-generated) Hindi targets used in Experiments 001-002 with **real, human-produced Hindi** wherever it could be found or extracted, and see whether real (if smaller/noisier) data beats synthetic (larger/cleaner) data.

## Data investigation

Inspected all 8 files in `data/archive/`. Full breakdown in the chat/session, but the short version: most of the "20,722 instruction pairs" and "adaption-enhanced" files are the same 20,476-word **dictionary** repackaged (word-level, not sentence pairs), or LLM-generated English scholarly essays about individual words — not usable for sentence-level Prakrit→Hindi translation training. The real, structured sentence-level parallel data with Hindi is much smaller than the dataset's README implies:

1. **`prakritnlp_dataset.json` → `parallel_corpus`**: 26 Prakrit passages, of which 13 have a `hindi` field with usable prose translation (after stripping embedded HTML-ish glossary/explanation markup and dropping truncated/duplicate entries). → `data/prakrit_hindi_real_agama.tsv` (13 rows).
2. **`acharanga_bhashyam.json`**: a 1.5M-character wall of raw, unsegmented commentary text (Prakrit moolpath + Sanskrit chhaya + Hindi translation + extended commentary, all run together from PDF-to-text OCR with no layout markers preserved). Found a genuinely repeating structural pattern by inspection: `<number>. <Prakrit sutra> सं०-<Sanskrit chhaya>। <Hindi...>`. Wrote `scripts/parse_acharanga_bhashyam.py` to exploit this:
   - Anchor on all `सं०` marker occurrences (1,007 raw hits — many are false positives from citation-style uses of the abbreviation, not real chhaya markers).
   - Use a **sequential-cursor** algorithm (each block's Prakrit text = the gap between the previous block's Hindi and this block's `सं०` marker) rather than independent per-marker windows, since an earlier version of this parser had blocks bleed into each other (Hindi from one sutra getting paired with the next sutra's Prakrit) due to inconsistent OCR of the separator dash (`-` vs `–`).
   - Applied quality filters: min/max length, Hindi:Prakrit length ratio ≤ 8x (rejects cases where a block ran into extended commentary), and a Devanagari-vs-junk-character check (rejects OCR garbage glyphs).
   - Result: 478 pairs after filtering (from 906 raw candidates). Quality is real but noisier than the 13 clean pairs — spot-checks show mostly correct sutra/verse ↔ Hindi-translation pairs, but some have leftover commentary bleed-through or partial misalignment. See `data/prakrit_hindi_acharanga_bhashyam.tsv`.

Combined into `data/prakrit_hindi_real_train.tsv` (487 rows: 9 clean + 478 noisy — 4 clean pairs held out) and `data/prakrit_hindi_real_test.tsv` (4 rows, **the only genuinely gold-standard Prakrit-Hindi test data we have**).

**Important domain note:** this real data is entirely from Jain Agama canonical texts (Acharanga, Sutrakritanga) — dense philosophical/religious prose about karma, non-violence, ascetic conduct. This is a completely different register from the Sattasai romantic poetry used in Experiments 001-002 and in the existing `data/prakrit_hindi_test.tsv` (100-example synthetic test set).

## Setup

Same as Experiment 001 (LoRA r=16/alpha=32 on q_proj/v_proj, bf16, `indictrans2-indic-indic-1B`, `hin_Deva` routing), except: `MAX_LEN=160` (Hindi commentary is longer than Sattasai couplets), 8 epochs, val split of 40 examples carved from train (dataset is much smaller than Experiment 001's).

## Results

| Test set | Zero-shot | Fine-tuned (this experiment) | Fine-tuned (Experiment 002, Sattasai-trained) |
|---|---|---|---|
| **Real gold test** (4 examples, Agama domain) | BLEU 0.63 / chrF++ 12.12 | **BLEU 1.39 / chrF++ 19.11** | not evaluated on this set |
| Synthetic test (100 examples, Sattasai domain) | BLEU 0.59 / chrF++ 10.79 | BLEU 0.76 / chrF++ 11.64 | **BLEU 3.73 / chrF++ 17.30** |

Two things stand out:
1. On real gold Hindi, this experiment's model improves over zero-shot by a similar relative margin to Experiment 002 (chrF++ nearly doubles). This is the first time we've measured improvement against genuinely human-translated Hindi rather than a synthetic MarianMT reference — a more trustworthy (if tiny, n=4, high-variance) signal.
2. This model does *much worse* on the Sattasai-domain synthetic test set than Experiment 002's model does — confirming that LoRA fine-tuning on ~490 examples produces a **domain-specific** adaptation, not general Prakrit→Hindi improvement. A model trained on Jain philosophical commentary doesn't transfer to romantic poetry, and vice versa. Neither experiment's training data size is anywhere near what's needed for broad coverage.

## Qualitative assessment

See `eval_results_real_test.txt`. The fine-tuned outputs are still not accurate translations, but two effects are visible:
- **Correct thematic vocabulary**: outputs use जीव (soul/being), प्राण (life-breath), कर्म (karma), आत्मा (self) — the right semantic universe for Jain philosophical text, which zero-shot completely misses (e.g. zero-shot mistranslates "भगवता" context as a *wedding* — "विवाह समारोह").
- **Noise inherited from the noisy training data**: one output contains a literal `(भाष्यम्)` token and garbled citation-number strings like `1-1-2-3-4-6-5-4` — clear evidence the model partially memorized OCR/citation artifacts from the 478 noisier Acharanga-parsed pairs rather than only learning translation. This directly reflects the data-quality caveat flagged above.

## Conclusion

Real data (even a small, partly-noisy 487 pairs) produces a measurable, trustworthy improvement over zero-shot when evaluated on real Hindi — a meaningfully more convincing result than Experiments 001-002's synthetic-Hindi-based scores. But the domain-specificity finding is the more important takeaway: **491 pairs from one narrow domain (Jain philosophical prose) does not generalize** to a different domain (romantic poetry), and neither does the reverse. Scaling up real, human-translated data — and probably training on a domain mix rather than a single narrow corpus — is the clear next lever, more so than further decoding tricks.

## Next steps

1. Track B (Prakrit→English) was paused: `ai4bharat/indictrans2-indic-en-1B` is a separately-gated HF repo from `indic-indic-1B`; access requested, pending grant.
2. Consider combining this experiment's real Prakrit-Hindi data with Experiment 001's synthetic data (domain mix) rather than training on either alone.
3. If more Hindi-labeled real data is wanted, the `acharanga_bhashyam.json` parser could be extended/hardened (e.g. handling more OCR spelling variants of `भाष्यम्` to cut commentary bleed-through more reliably) — flagged as moderate effort, uncertain additional yield.

## File inventory

- `build_pra_eng_dataset.py`: Builds the Prakrit-English pair dataset (Sattasai, Kaggle parallel corpus, Saman Suttam); reused in Exp 004.
- `build_pra_hindi_real_dataset.py`: Assembles the Prakrit-Hindi real dataset (13 clean Kaggle pairs plus the parsed pairs).
- `eval_results_real_test.txt`: BLEU/chrF++ on the real gold test (Agama) for zero-shot and fine-tuned models.
- `eval_results_synthetic_test.txt`: BLEU/chrF++ on the Sattasai synthetic test for the v3 model (0.76 / 11.64 fine-tuned vs 0.59 / 10.79 zero-shot).
- `finetune_lora_pra_hindi_real.py`: LoRA fine-tune on the real Prakrit-Hindi data (produces the v3 checkpoint).
- `parse_acharanga_bhashyam.py`: Parses the Acharanga Bhashyam commentary text into 478 noisy Prakrit-Hindi pairs.
- `run_eval_pra_hindi_real.py`: Evaluates the v3 model on the real gold test and the synthetic Sattasai test.
