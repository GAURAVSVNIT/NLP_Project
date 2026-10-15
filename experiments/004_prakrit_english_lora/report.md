# Experiment 004: Prakrit→English LoRA fine-tune (natively-supported direction, real data)

**Date:** 2026-09-06
**Model checkpoint:** `models/checkpoints/v4_prakrit_english_real_data/`
**Goal:** Test the hypothesis (flagged in Experiment 003's next-steps) that Prakrit→English — using IndicTrans2's natively-supported English target (no `hin_Deva` routing hack needed on the target side) plus more real training data than we'd previously used — would outperform the Prakrit→Hindi track by a wide margin.

## Setup

- Model: `ai4bharat/indictrans2-indic-en-1B` (separately-gated HF repo from `indic-indic-1B`; access requested and granted mid-session).
- Routing: source Prakrit tagged `hin_Deva` (still needed, no native Prakrit tag), target tagged `eng_Latn` — this is a **real**, natively-supported target language for IndicTrans2, not a routing hack.
- LoRA: same as prior experiments (r=16, alpha=32, q_proj/v_proj, bf16).
- Data: `data/prakrit_english_combined.tsv` — 1,334 real, human-translated Prakrit-English pairs (1,299 existing Sattasai pairs + 20 Kaggle `parallel_corpus` + 230 Kaggle Saman Suttam-with-English, deduped). 100 held out for validation during training; the pre-existing 100-line `data/100_test_pra.txt`/`100_test_eng.txt` (confirmed zero overlap with training data) used as the real test set.
- 6 epochs, effective batch size 16, max length 128.

## Results

| | BLEU | chrF++ |
|---|---|---|
| Zero-shot baseline | 2.14 | 13.76 |
| **LoRA fine-tuned** | **9.81** (4.6x) | **25.30** (1.8x) |

By far the strongest result across all four experiments — both the absolute BLEU/chrF++ and the relative improvement over zero-shot dwarf anything achieved on the Prakrit→Hindi track (best previous fine-tuned BLEU was 3.73, Experiment 002).

Training loss trajectory was also much healthier in absolute terms than the Hindi experiments: eval loss 3.72 → 3.49 across 6 epochs, versus 5.5-6.3 range for the Hindi real-data run — consistent with English being a language the base model already represents very well, versus Hindi requiring the `hin_Deva`-for-Prakrit routing hack on both ends.

## Qualitative assessment

See `eval_results.txt`. This is a real, visible qualitative jump, not just a metric artifact:
- **Zero-shot** frequently fails outright — either transliterating Prakrit into English letters untranslated (`"Ajjaon Chea Potho Ujagaro Janas Ajja..."`), or producing fluent English that's thematically unrelated to the source (`"you'll also want to have a water cage"`, `"you will not be able to go to the Kumbh Mela"`).
- **Fine-tuned** consistently produces complete, grammatical English sentences that land in the correct thematic register for Hala's Sattasai — romantic/erotic classical Indian poetry about lovers, husbands, wives, bodies, longing (`"her husband's face"`, `"eyes glistening with tears"`, `"her lips clung to her breasts"`, `"you don't know how to make love"`). Word-for-word accuracy against the reference is still low, but the model has clearly learned the domain's vocabulary and imagery, not just fluent-but-arbitrary Hindi/English the way earlier Hindi-direction experiments did.

## Conclusion

This confirms the strategic direction change flagged after the Kaggle data investigation: for this project's data (dominated by Sattasai-style Prakrit-English pairs, since real Prakrit-Hindi data is scarce and narrow-domain), fine-tuning the **natively-supported Prakrit→English direction** is a substantially better use of effort than continuing to push the `hin_Deva`-routed Prakrit→Hindi direction. This is now the strongest baseline in the project.

## Next steps

1. Try combining this Prakrit-English data with the Kaggle `saman_suttam` and `parallel_corpus` monolingual/instruction data more aggressively, or add continued pretraining on the 97K-line monolingual Prakrit corpus, to push BLEU further.
2. Revisit Prakrit→Hindi only if more real, broad-domain Hindi parallel data becomes available (Experiment 003 showed 491 narrow-domain pairs isn't enough to generalize).
3. Consider evaluating with COMET (semantic similarity) alongside BLEU/chrF++, since these poetry translations are often semantically-reasonable paraphrases that surface-level n-gram metrics penalize harshly.

## File inventory

- `build_pra_eng_dataset.py`: Builds the 1,334 Prakrit-English pairs used for training.
- `eval_results.txt`: BLEU/chrF++ and sample outputs for the English direction.
- `finetune_lora_pra_eng.py`: LoRA fine-tune of `indictrans2-indic-en-1B` on Prakrit-English pairs (produces the v4 checkpoint).
- `run_eval_pra_eng.py`: Evaluates the v4 model on the 100-line Prakrit-English test.
