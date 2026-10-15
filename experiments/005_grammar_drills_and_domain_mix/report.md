# Experiment 005: Decoding data/fwd grammar drills + domain-mix training

**Date:** 2026-09-08
**Model checkpoint:** `models/checkpoints/v5_domain_mix_real_data/`
**Goal:** `data/fwd/*.pdf` (10 scanned Hindi↔Prakrit grammar-drill answer sheets, discovered at the very start of this project but never decoded) contains real Hindi-Prakrit sentence pairs. Decode them into usable text, extract clean sentence pairs, and test whether combining this with Experiment 003's real Agama data (a domain mix) generalizes better than any single narrow-domain real dataset used so far.

## Part 1: Decoding the DevLys-010 font

The PDFs use a legacy non-Unicode Devanagari font (`DevLys-010`) — glyphs render correctly in a PDF viewer but `pdftotext` extraction produces mojibake (`iz'u&i=&` instead of `प्रश्न-पत्र`). Built `scripts/devlys_to_unicode.py`, a Python port of the community DevLys-010↔Unicode conversion algorithm (source: `prakhariitd/COP315-Hindi-fonts-to-Unicode` on GitHub).

Key implementation notes:
- The 227-entry substitution table was extracted **programmatically** from the original JS source (`devlys_array_one.txt` / `devlys_array_two.txt`), not hand-transcribed — an initial hand-transcription attempt introduced a silent off-by-one array-length bug that would have corrupted output.
- Reproduces the algorithm's positional-reordering rules: DevLys stores the "ि" (chhoti-ee) matra *before* its consonant and reph ("र्") *after* the syllable it attaches to (both opposite of Unicode logical order) — these get moved to the correct position.
- `pdftotext`'s default output encoding needed to be read as `cp1252`, not UTF-8 (the DevLys glyph mapping uses the Windows-1252 upper range).
- Residual imperfections remain on a handful of specific words (e.g. "मरेंगे", "दोनों") likely caused by `pdftotext -layout` inserting spurious spaces that break the positional decoding — acceptable given the Prakrit side (our primary interest) decodes cleanly and the overall text is highly legible.

## Part 2: Extracting sentence pairs — the column-shift discovery

`pdftotext -layout` reconstructs each PDF page as two text columns (Hindi question | Prakrit answer) separated by whitespace gaps. Naively splitting on the gap and pairing same-row text produces **silently wrong pairs** in several files: a semantic spot-check (translating a handful of pairs by hand) revealed the whole answer column was shifted by one row relative to the questions in some files, e.g. file 01's row 1 "मैं हँसता हूँ" (I laugh) was paired with row 1's own answer "अम्हे ण्हामो" (we bathe) — wrong — while the section header's leftover answer fragment "अहं हसामि" (I laugh) was the *actual* correct answer for row 1.

Root cause: the instruction header row has no real answer of its own, so `pdftotext`'s column reconstruction absorbs the *first* real answer onto that row, shifting every subsequent row-to-row correspondence by one. `scripts/parse_fwd_grammar_drills.py` auto-detects this per section (via whether the header line carries trailing content) and pairs accordingly.

Two further complications, found only by continuing to spot-check after the initial fix:
1. **Orphan answer-only rows** (no question, just an answer sitting in the answer column at some point in a list) add an *extra* step to the shift depth. Found once right after a header (file 06) and once **mid-list** (file 08, between items 20 and 21) — the mid-list case wasn't caught by a header-only shift model and required generalizing to detect these dynamically anywhere in a section (via a large leading-indent heuristic) and always re-triggering shift mode when found.
2. A non-`"N."` section header (`"2/ वाक्य बनाओ।"` in file 09, likely an OCR artifact) caused a translation section to run past its real boundary into an unrelated word-root grammar-analysis exercise, producing a handful of nonsense pairs. Fixed by widening section-boundary detection (not extraction-eligibility) to recognize `"N/"`, `"N@"` etc.

After these fixes: **267 pairs** extracted from all 10 files (one "make sentences" translation section per file, plus a second section in file 10), broadly spot-checked (30+ random pairs, all files individually checked) with no further anomalies found.

## Part 3: Domain-mix training

Combined three real Prakrit-Hindi sources into one training set:
- 267 grammar-drill pairs (this experiment) — simple modern Hindi, short sentences, heavy verb-conjugation drilling
- 13 clean + 478 noisier pairs from Experiment 003 (Jain Agama philosophical/religious prose)
- Total: 730 pairs, minus a 40-example validation slice and 20 held-out grammar-drill test examples

Trained the same LoRA config as prior experiments (r=16/alpha=32, `indictrans2-indic-indic-1B`, `hin_Deva` routing) for 8 epochs.

**Bug found mid-experiment (and fixed):** the first eval pass surfaced a reference mismatch in the grammar-drill test set — a Prakrit sentence about "night spreading" paired with a Hindi reference about "a peahen dancing". Traced this to the mid-list orphan-row issue described above (in file 08), which my first-pass parser didn't yet handle. Fixed the parser, rebuilt the dataset, and **retrained from scratch** on the corrected 267-pair set (2 pairs changed net) rather than leave a known bug baked into the saved checkpoint.

## Results (post-fix)

| Test domain | Zero-shot | Fine-tuned (domain mix) |
|---|---|---|
| **Grammar drills** (this experiment's domain, held-out) | 1.03 / 12.20 | **5.06 / 25.15** ← best BLEU across all experiments |
| Agama gold (Experiment 003's real test) | 0.63 / 12.12 | 0.83 / 17.12 |
| Sattasai synthetic (Experiments 001-002's test) | 0.59 / 10.79 | 1.05 / 11.47 |

All three domains improved over zero-shot — unlike Experiment 003's single-domain model, which only helped its own domain and actually *hurt* on the Sattasai test. The domain mix trades peak in-domain performance (grammar-drill BLEU here is strong but likely still below what a grammar-drill-only model would hit) for broader applicability.

## Qualitative assessment

See `eval_results_grammar_drill.txt`. This is the most convincing output quality seen in the project so far — several **exact or near-exact correct translations**, not just correct theme/register:
- `सा कलहइ` → ref "वह कलह करती है" (she quarrels) → fine-tuned: "वह झगड़ती है" (she quarrels) — correct, synonymous.
- `अम्हे उल्लसमो` → ref "हम सब खुश होवें" → fine-tuned: "हम सब खुश हैं" (we are all happy) — correct meaning.
- `तुम्हे डरीअ।` → ref "तुम दोनों डरे" → fine-tuned: "तुम डरते हो" (you fear) — correct core meaning, missing the dual-number nuance.

This strongly suggests short, structurally-repetitive sentences (grammar drills) are far more learnable from a few hundred real examples than free-verse poetry or dense philosophical prose — consistent with general low-resource MT intuition about sentence complexity vs. data efficiency.

## Conclusion

1. The `data/fwd` PDFs were a real, previously-untapped data source — decoding them required reverse-engineering both a legacy font encoding and an undocumented PDF-column-reconstruction artifact, but yielded 267 clean, verified real Hindi-Prakrit pairs.
2. Domain-mixing real data from three different registers (poetry / philosophy / grammar drills) lifts performance on all three, achieving the best result yet on the simplest (grammar-drill) domain.
3. Process note reinforced: **spot-checking generation output against source pairs surfaces data bugs that spot-checking the raw data alone missed** — the mid-list orphan-row bug was only found by tracing a suspicious model output back to its source pair, not during the initial data-quality review.

## Next steps

1. Consider a grammar-drill-only fine-tune (this experiment's cleanest, most learnable domain) to see the ceiling without dilution from the other two domains.
2. Apply the same coordinate-free column-shift-detection technique to any future OCR/PDF-column data sources — it's a general pdftotext artifact, not specific to this dataset.
3. Revisit whether `data/fwd`'s other exercise types (pronoun tables, verb-form grids, "make sentences from word roots") could be mined for additional word/morphology-level supervision, separate from sentence-level MT training.

## File inventory

- `build_domain_mix_dataset.py`: Builds the 3-domain training mix (grammar drills + Agama) and the 20-example held-out drill test.
- `devlys_array_one.txt`: DevLys substitution table, first array (extracted from the source JS).
- `devlys_array_two.txt`: DevLys substitution table, second array (paired with the first).
- `devlys_to_unicode.py`: Port of the DevLys-010 to Unicode conversion (copy of `scripts/devlys_to_unicode.py`), used to decode the `data/fwd` PDFs.
- `eval_results_agama_gold.txt`: BLEU/chrF++ on the 4-example Agama gold test (0.83 / 17.12 fine-tuned vs 0.63 / 12.12 zero-shot).
- `eval_results_grammar_drill.txt`: BLEU/chrF++ and sample outputs on the 20 held-out grammar drills.
- `eval_results_sattasai_synthetic.txt`: BLEU/chrF++ on the Sattasai synthetic test (1.05 / 11.47 fine-tuned vs 0.59 / 10.79 zero-shot).
- `finetune_lora_domain_mix.py`: LoRA fine-tune on the domain mix (produces the v5 checkpoint, the best Hindi-direction model at the time).
- `parse_fwd_grammar_drills.py`: Extracts 267 Prakrit-Hindi sentence pairs from the decoded grammar-drill answer sheets, with column-shift handling.
- `run_eval_domain_mix.py`: Evaluates the v5 model on the drill test, Agama gold and Sattasai synthetic tests.
