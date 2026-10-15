# Experiment 006: Vajjalagga gatha pairs from `word फाइल.docx`

**Date:** 2026-10-09
**Status:** data extraction only. No model trained or evaluated.
**Output:** `data/prakrit_hindi_vajjalagga.tsv` (20 Prakrit→Hindi pairs)
**Script:** `extract_vajjalagga_pairs.py`

**Goal:** `word फाइल.docx` (edited today, not yet used) is a Vajjalagga text: numbered Prakrit gathas, each followed by a Hindi `अर्थ-` gloss. Extract the gatha→gloss pairs as real Prakrit→Hindi data, in the same two-column TSV format as the other `data/prakrit_hindi_*.tsv` files.

## Decoding

The document uses the same legacy Devanagari family as the `data/fwd` PDFs. The existing `scripts/devlys_to_unicode.py` decodes it cleanly without changes. Checked on the first lines, e.g. `oTtkyXx ¼p;fur va'k½` → `वज्जालग्ग (चयनित अंश)`, and `ikb;dOoLl` → `पाइयकव्वस्स`. Converter output was read against the source for the first verses before the parser was written.

## Structure

- Paragraphs 001–059: 20 gathas, each `N. <line 1>` / `<line 2> ।। (M)` followed by one `अर्थ- …` paragraph. Verse numbers in the text run 1–20 with no gaps.
- Paragraphs 061 onward: a Prakrit prose story (`अमंगलियपुरिसस्स कहा`) with a Hindi translation (`अमांगलिक पुरुष की कथा`) and a second story (`घर में शूर`). **Not extracted.** The Prakrit and Hindi paragraphs do not line up one-to-one (one Hindi paragraph covers part of one Prakrit paragraph and all of the next), so pairing them needs a sentence aligner.

## Parser checks

- 20 gathas → 20 pairs. 0 orphan glosses, 0 unterminated verses.
- No empty fields; every row has exactly 2 columns.
- Spot-checked against the decoded source: pair 1 is the invocation verse with its gloss, pair 3 (सुयणो न कुप्पइ…) and pair 20 (झीणविहवो…) match their glosses.

## Caveats

- **20 pairs is small.** It is far below the ~730 PR-HI pairs in Exp 005. It is useful as a held-out check or for a small domain test, not as training data on its own.
- **Domain:** these are gnomic/ethical verses (sajjana, kshama, etc.), which is a different register from the grammar-drill and Agama domains.
- The glosses are free Hindi paraphrase, often expanding the verse. They are not word-for-word, so BLEU against them will look low even for good output. chrF++ and manual reading are the better measures.

## Next steps

1. Decide whether these 20 pairs are a test set (keep out of training) or training data (add to the Exp 005 mix and retrain). A test set is the cleaner choice, since the Exp 005 comparison would otherwise change.
2. Align the prose-story paragraphs with a sentence-level aligner, or by hand, to get more pairs from the same document.

## File inventory

- `extract_vajjalagga_pairs.py`: Decodes `word फाइल.docx` and extracts the 20 gatha-to-Hindi-gloss pairs into `data/prakrit_hindi_vajjalagga.tsv`.

## Artifacts outside this folder

- `data/prakrit_hindi_vajjalagga.tsv`: the 20 extracted pairs (the output of the extractor).
