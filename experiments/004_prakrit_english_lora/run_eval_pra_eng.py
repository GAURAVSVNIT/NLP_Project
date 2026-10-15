import csv
import sys

import evaluate
import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

MODEL_PATH = "models/indictrans2-indic-en-1B"
LORA_PATH = "models/checkpoints/v4_prakrit_english_real_data"
SRC_TAG = "hin_Deva"
TGT_TAG = "eng_Latn"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
MAX_LEN = 128
OUT_DIR = "experiments/004_prakrit_english_lora"


def load_test():
    with open("data/100_test_pra.txt", encoding="utf-8") as f:
        prakrit = [l.strip() for l in f if l.strip()]
    with open("data/100_test_eng.txt", encoding="utf-8") as f:
        english = [l.strip() for l in f if l.strip()]
    assert len(prakrit) == len(english)
    return prakrit, english


def translate(prakrit_sentences, tokenizer, model, ip):
    outputs = []
    for i in range(0, len(prakrit_sentences), BATCH_SIZE):
        batch = prakrit_sentences[i : i + BATCH_SIZE]
        pre = ip.preprocess_batch(batch, src_lang=SRC_TAG, tgt_lang=TGT_TAG)
        inputs = tokenizer(pre, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                num_beams=5,
                max_length=MAX_LEN,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
            )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        post = ip.postprocess_batch(decoded, lang=TGT_TAG)
        outputs.extend(post)
        print(f"  {min(i + BATCH_SIZE, len(prakrit_sentences))}/{len(prakrit_sentences)}", file=sys.stderr)
        del inputs, generated
        torch.cuda.empty_cache()
    return outputs


def score(hyps, refs):
    bleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")
    bleu_score = bleu.compute(predictions=hyps, references=[[r] for r in refs])
    chrf_score = chrf.compute(predictions=hyps, references=[[r] for r in refs], word_order=2)
    return bleu_score["score"], chrf_score["score"]


def main():
    prakrit, english_ref = load_test()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    ip = IndicProcessor(inference=True)

    print("Loading base model for zero-shot baseline...", file=sys.stderr)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float16).to(DEVICE)
    base_model.eval()
    zero_shot_out = translate(prakrit, tokenizer, base_model, ip)
    zs_bleu, zs_chrf = score(zero_shot_out, english_ref)
    print(f"zero-shot: BLEU={zs_bleu:.2f} chrF++={zs_chrf:.2f}", file=sys.stderr)
    del base_model
    torch.cuda.empty_cache()

    print("Loading LoRA fine-tuned model...", file=sys.stderr)
    base_model2 = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float16).to(DEVICE)
    ft_model = PeftModel.from_pretrained(base_model2, LORA_PATH).to(DEVICE)
    ft_model.eval()
    ft_out = translate(prakrit, tokenizer, ft_model, ip)
    ft_bleu, ft_chrf = score(ft_out, english_ref)
    print(f"fine-tuned: BLEU={ft_bleu:.2f} chrF++={ft_chrf:.2f}", file=sys.stderr)
    del base_model2, ft_model
    torch.cuda.empty_cache()

    with open(f"{OUT_DIR}/eval_results.txt", "w", encoding="utf-8") as f:
        f.write(f"Test set: data/100_test_pra.txt + 100_test_eng.txt ({len(prakrit)} examples)\n")
        f.write(f"Zero-shot baseline: BLEU={zs_bleu:.2f} chrF++={zs_chrf:.2f}\n")
        f.write(f"LoRA fine-tuned:    BLEU={ft_bleu:.2f} chrF++={ft_chrf:.2f}\n\n")
        f.write("Sample comparisons (first 15):\n")
        for i in range(min(15, len(prakrit))):
            f.write(f"\n[{i}] Prakrit:    {prakrit[i]}\n")
            f.write(f"    English ref: {english_ref[i]}\n")
            f.write(f"    Zero-shot:   {zero_shot_out[i]}\n")
            f.write(f"    Fine-tuned:  {ft_out[i]}\n")
    print(f"Wrote {OUT_DIR}/eval_results.txt", file=sys.stderr)


if __name__ == "__main__":
    main()
