import csv
import sys

import evaluate
import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

MODEL_PATH = "models/indictrans2-indic-indic-1B"
LORA_PATH = "models/checkpoints/v1_synthetic_hindi_baseline"
LANG_TAG = "hin_Deva"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16


def load_test():
    rows = []
    with open("data/prakrit_hindi_test.tsv", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append((row["prakrit"], row["hindi"]))
    return rows


def translate(prakrit_sentences, tokenizer, model, ip):
    outputs = []
    for i in range(0, len(prakrit_sentences), BATCH_SIZE):
        batch = prakrit_sentences[i : i + BATCH_SIZE]
        pre = ip.preprocess_batch(batch, src_lang=LANG_TAG, tgt_lang=LANG_TAG)
        inputs = tokenizer(pre, padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                num_beams=5,
                max_length=128,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
            )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        post = ip.postprocess_batch(decoded, lang=LANG_TAG)
        outputs.extend(post)
        print(f"  {min(i + BATCH_SIZE, len(prakrit_sentences))}/{len(prakrit_sentences)}", file=sys.stderr)
    return outputs


def score(hyps, refs):
    bleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")
    bleu_score = bleu.compute(predictions=hyps, references=[[r] for r in refs])
    chrf_score = chrf.compute(predictions=hyps, references=[[r] for r in refs], word_order=2)
    return bleu_score["score"], chrf_score["score"]


def main():
    test_rows = load_test()
    prakrit = [r[0] for r in test_rows]
    hindi_ref = [r[1] for r in test_rows]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    ip = IndicProcessor(inference=True)

    print("Loading base model for zero-shot baseline...", file=sys.stderr)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float16).to(DEVICE)
    base_model.eval()
    zero_shot_out = translate(prakrit, tokenizer, base_model, ip)
    zs_bleu, zs_chrf = score(zero_shot_out, hindi_ref)

    print("Loading LoRA fine-tuned model...", file=sys.stderr)
    del base_model
    torch.cuda.empty_cache()
    base_model2 = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float16).to(DEVICE)
    ft_model = PeftModel.from_pretrained(base_model2, LORA_PATH).to(DEVICE)
    ft_model.eval()
    ft_out = translate(prakrit, tokenizer, ft_model, ip)
    ft_bleu, ft_chrf = score(ft_out, hindi_ref)

    with open("experiments/002_repetition_penalty_decoding/eval_results.txt", "w", encoding="utf-8") as f:
        f.write(f"Zero-shot baseline: BLEU={zs_bleu:.2f} chrF++={zs_chrf:.2f}\n")
        f.write(f"LoRA fine-tuned:    BLEU={ft_bleu:.2f} chrF++={ft_chrf:.2f}\n\n")
        f.write("Sample comparisons (first 15):\n")
        for i in range(min(15, len(prakrit))):
            f.write(f"\n[{i}] Prakrit:    {prakrit[i]}\n")
            f.write(f"    Hindi ref:  {hindi_ref[i]}\n")
            f.write(f"    Zero-shot:  {zero_shot_out[i]}\n")
            f.write(f"    Fine-tuned: {ft_out[i]}\n")

    print("Wrote experiments/002_repetition_penalty_decoding/eval_results.txt", file=sys.stderr)
    print(f"Zero-shot: BLEU={zs_bleu:.2f} chrF++={zs_chrf:.2f}", file=sys.stderr)
    print(f"Fine-tuned: BLEU={ft_bleu:.2f} chrF++={ft_chrf:.2f}", file=sys.stderr)


if __name__ == "__main__":
    main()
