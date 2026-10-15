# Experiment 010: Narrative test set from the Vajjalagga prose stories

**Date:** 2026-10-13
**Source:** `word फाइल.docx`, the two prose stories (अमंगलिक पुरुष की कथा, घर में शूर)
**Output:** `data/prakrit_hindi_narrative_test.tsv` (47 hand-checked pairs), with all 56 automatic blocks in `data/prakrit_hindi_narrative_aligned.tsv`

**Question:** Do the existing models generalize to narrative prose, a register not in any current test set?

## Alignment

Paragraph counts differ between the Prakrit and Hindi sides, so the alignment is sentence-level. Both sides are split on danda and question marks, then aligned with a length-based dynamic program (Gale and Church style), allowing 1-1, 1-2, 2-1 and 2-2 blocks.

The automatic alignment produced 56 blocks. I read all of them against the source text and dropped:
- **Block 33:** a goldsmith-walking line paired with a wife-returns-home line.
- **Blocks 48-55:** from the middle of the second story to its end. Here the alignment is shifted by about one block, so each Prakrit sentence is paired with the next Hindi sentence.

Those exclusions are listed in the script with their reasons. The remaining 47 blocks were checked by reading and match in meaning. Blocks are sentence groups, not always single sentences.

## Leakage check

None of the 47 Prakrit sides appears exactly in any training file. Overlap is 0.

## Results

Decoding settings match Exp 005 and 007 (beam 5, repetition_penalty 1.3, no_repeat_ngram_size 3).

| System | BLEU | chrF++ |
|---|---|---|
| Zero-shot IndicTrans2 | 0.75 | 13.69 |
| Domain mix (Exp 005, v5) | 0.66 | 14.92 |
| Drill-only (Exp 007, v6) | 0.69 | 12.73 |

## Interpretation

- **No model generalizes to narrative prose.** All three score under 1 BLEU. The fine-tuned models do not beat zero-shot on BLEU.
- **The domain mix is the only one above zero-shot on chrF++** (+1.2), and drill-only is below zero-shot (-1.0). This is weak evidence that the broader training mix transfers slightly better, but with 47 pairs and paraphrased references the gap is within noise.
- **Reference style limits the scores.** The Hindi is a free narrative translation that adds and drops words, so n-gram overlap is low even for a correct translation.

## Limitations

- 47 pairs, from one document, in one story style.
- The alignment is partly manual (the exclusions), so the set is not fully automatic.
- Verse glosses (Exp 006) are not included here.

## File inventory

- `align_story_pairs.py`: extracts the story paragraphs, splits them into sentences, aligns them, applies the hand-checked exclusions, and writes both TSV files.
- `run_eval_narrative.py`: runs the three systems on the 47 pairs and checks for leakage. Writes `eval_results.txt`.
- `eval_results.txt`: the scores above.
- `eval_log.txt`: the run log.
