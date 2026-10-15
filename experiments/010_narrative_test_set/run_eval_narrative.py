"""
Experiment 010 eval: score zero-shot, domain-mix (v5) and drill-only (v6) on the
47 hand-checked narrative pairs. Also checks that no test sentence appears in any
training file. Decoding settings match Exp 005/007 (beam 5, repetition_penalty
1.3, no_repeat_ngram_size 3).

Usage (from project root): python experiments/010_narrative_test_set/run_eval_narrative.py
"""
import csv
import sys
from pathlib import Path

import evaluate
import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

MODEL_PATH = "models/indictrans2-indic-indic-1B"
LANG_TAG = "hin_Deva"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST = Path("data/prakrit_hindi_narrative_test.tsv")
OUT = Path("experiments/010_narrative_test_set/eval_results.txt")
SYSTEMS = [
    ("zero_shot", None),
    ("domain_mix_v5", "models/checkpoints/v5_domain_mix_real_data"),
    ("drill_only_v6", "models/checkpoints/v6_grammar_drill_only"),
]
TRAIN_FILES = [
    "data/prakrit_hindi_domain_mix_train.tsv",
    "data/prakrit_hindi_drill_only_train.tsv",
    "data/prakrit_hindi_grammar_drills.tsv",
    "data/prakrit_hindi_real_agama.tsv",
    "data/prakrit_hindi_acharanga_bhashyam.tsv",
    "data/prakrit_hindi_train.tsv",
]


def load(path):
    with open(path, encoding="utf-8") as f:
        return [(r["prakrit"], r["hindi"]) for r in csv.DictReader(f, delimiter="\t")]


def translate(model, tok, ip, sents):
    out = []
    for i in range(0, len(sents), 8):
        pre = ip.preprocess_batch(sents[i:i + 8], src_lang=LANG_TAG, tgt_lang=LANG_TAG)
        inp = tok(pre, padding=True, truncation=True, max_length=160, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            gen = model.generate(**inp, num_beams=5, max_length=160, repetition_penalty=1.3, no_repeat_ngram_size=3)
        out += ip.postprocess_batch(tok.batch_decode(gen, skip_special_tokens=True), lang=LANG_TAG)
    return out


def main():
    test = load(TEST)
    src = [p for p, _ in test]
    refs = [h for _, h in test]

    # leakage check: exact sentence overlap with any training file
    train_src = set()
    for f in TRAIN_FILES:
        if Path(f).exists():
            train_src |= {p for p, _ in load(f)}
    overlap = sum(1 for p in src if p in train_src)
    print(f"test sentences: {len(src)}, exact overlap with training sources: {overlap}", file=sys.stderr)

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    ip = IndicProcessor(inference=True)
    bleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")
    lines = [f"test: {TEST} ({len(src)} pairs), exact overlap with training sources: {overlap}", ""]

    for name, lora in SYSTEMS:
        base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                                     torch_dtype=torch.float16).to(DEVICE)
        model = PeftModel.from_pretrained(base, lora).to(DEVICE) if lora else base
        model.eval()
        hyps = translate(model, tok, ip, src)
        b = bleu.compute(predictions=hyps, references=[[r] for r in refs])["score"]
        c = chrf.compute(predictions=hyps, references=[[r] for r in refs], word_order=2)["score"]
        lines.append(f"{name}: BLEU={b:.2f} chrF++={c:.2f}")
        print(f"{name}: BLEU={b:.2f} chrF++={c:.2f}", file=sys.stderr)
        del model, base
        torch.cuda.empty_cache()

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
