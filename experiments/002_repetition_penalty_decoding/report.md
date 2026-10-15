# Experiment 002: Repetition-penalty decoding fix

**Date:** 2026-09-06
**Model checkpoint:** same as Experiment 001, `models/checkpoints/v1_synthetic_hindi_baseline/` (no retraining — decoding-only change)
**Goal:** Experiment 001's fine-tuned model kept falling into repetition loops during beam search (e.g. `अपने पति के साथ, अपने पति के साथ...` repeated 20+ times), which was suspected to be masking the real BLEU/chrF signal underneath.

## Change

Added to `model.generate()` in `scripts/run_eval.py` for both the zero-shot baseline and the fine-tuned model:
- `repetition_penalty=1.3`
- `no_repeat_ngram_size=3`

No retraining involved — this is purely a decoding-time change, applied identically to both models for a fair comparison.

## Results

| | BLEU | chrF++ | vs. Experiment 001 (no rep. penalty) |
|---|---|---|---|
| Zero-shot baseline | 0.62 | 10.87 | BLEU 0.45→0.62, chrF++ 8.31→10.87 |
| LoRA fine-tuned | **3.73** | **17.30** | BLEU 0.56→3.73 (6.6x), chrF++ 8.64→17.30 (2x) |

The fine-tuned model's score moved far more than the zero-shot baseline's did, confirming the repetition loops in Experiment 001 were substantially masking real learned signal from the LoRA fine-tune — the model had learned something useful, but beam search's tendency to loop on a locally-confident phrase was drowning it out in the n-gram overlap metrics.

## Qualitative assessment

See `eval_results.txt`. Outputs are now complete, grammatical, non-repeating Hindi sentences for both models. Several fine-tuned outputs land in the correct thematic register for the source material (Hala's Sattasai — romantic/domestic poetry about lovers, husbands, wives) even when not accurate word-for-word translations, e.g.:

- `[2]` ref: *"आशा एक व्यक्ति को प्रेम से अलग करने की आग उत्पन्न करती है"* → fine-tuned: *"एक आदमी के रूप में...जो अपने आप को अलग करने के लिए तैयार है, वह मेरे लिए मृत्यु का कारण बन सकता है"* — wrong specifics, but same separation/love/death theme.
- `[4]` fine-tuned: *"जब वह अपनी आँखों से बाहर निकलती है, तो वह अपने पति के सामने झुक जाती है..."* — coherent domestic-scene Hindi, plausible register even though not matching the reference content.

Translations are still not semantically accurate/usable — this decoding fix does not address the underlying lexical gap (the model doesn't yet know real Prakrit↔Hindi word mappings), only removed the beam-search degeneracy that was hiding the model's actual (still limited) progress.

## Conclusion

Cheap, worthwhile fix — always use `repetition_penalty`/`no_repeat_ngram_size` for this model going forward when evaluating, since beam search on an undertrained low-resource model defaults to looping. Confirms the fine-tuning in Experiment 001 has genuine (if still small) value. Does not change the core conclusion from Experiment 001: more/better real training data (PrakritNLP Kaggle corpus) is needed for translation quality to become usable.

## Next steps

1. Acquire the PrakritNLP Kaggle dataset (real curated Hindi/Sanskrit-Prakrit pairs + monolingual corpus).
2. Apply the same repetition-penalty decoding as the default in future eval runs (already updated in `scripts/run_eval.py`).

## File inventory

- `eval_results.txt`: BLEU/chrF++ and sample outputs for the decoding variants.
- `run_eval.py`: Decoding-only evaluation of the 001 model with `repetition_penalty=1.3` and `no_repeat_ngram_size=3`. No training.
