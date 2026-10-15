import csv
import sys

import evaluate
import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

MODEL_PATH = "models/indictrans2-indic-indic-1B"
LORA_PATH = "models/checkpoints/v3_real_kaggle_agama_data"
LANG_TAG = "hin_Deva"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
MAX_LEN = 160
OUT_DIR = "experiments/003_prakrit_hindi_real_kaggle_data"

TEST_SETS = [
    ("data/prakrit_hindi_real_test.tsv", "real"),
    ("data/prakrit_hindi_test.tsv", "synthetic"),
]


def load_test(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append((row["prakrit"], row["hindi"]))
    return rows


def translate(prakrit_sentences, tokenizer, model, ip):
    outputs = []
    for i in range(0, len(prakrit_sentences), BATCH_SIZE):
        batch = prakrit_sentences[i : i + BATCH_SIZE]
        pre = ip.preprocess_batch(batch, src_lang=LANG_TAG, tgt_lang=LANG_TAG)
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
        post = ip.postprocess_batch(decoded, lang=LANG_TAG)
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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    ip = IndicProcessor(inference=True)

    test_data = {path: load_test(path) for path, _ in TEST_SETS}

    results = {}  # (label, model_kind) -> (bleu, chrf, outputs)

    print("Loading base model for zero-shot baseline...", file=sys.stderr)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float16).to(DEVICE)
    base_model.eval()
    for path, label in TEST_SETS:
        rows = test_data[path]
        prakrit = [r[0] for r in rows]
        hindi_ref = [r[1] for r in rows]
        print(f"=== zero-shot on {label} ({len(rows)} examples) ===", file=sys.stderr)
        out = translate(prakrit, tokenizer, base_model, ip)
        bleu, chrf = score(out, hindi_ref)
        results[(label, "zero_shot")] = (bleu, chrf, out)
        print(f"zero-shot {label}: BLEU={bleu:.2f} chrF++={chrf:.2f}", file=sys.stderr)
    del base_model
    torch.cuda.empty_cache()

    print("Loading LoRA fine-tuned model (real Kaggle/Agama data)...", file=sys.stderr)
    base_model2 = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float16).to(DEVICE)
    ft_model = PeftModel.from_pretrained(base_model2, LORA_PATH).to(DEVICE)
    ft_model.eval()
    for path, label in TEST_SETS:
        rows = test_data[path]
        prakrit = [r[0] for r in rows]
        hindi_ref = [r[1] for r in rows]
        print(f"=== fine-tuned on {label} ({len(rows)} examples) ===", file=sys.stderr)
        out = translate(prakrit, tokenizer, ft_model, ip)
        bleu, chrf = score(out, hindi_ref)
        results[(label, "fine_tuned")] = (bleu, chrf, out)
        print(f"fine-tuned {label}: BLEU={bleu:.2f} chrF++={chrf:.2f}", file=sys.stderr)
    del base_model2, ft_model
    torch.cuda.empty_cache()

    for path, label in TEST_SETS:
        rows = test_data[path]
        prakrit = [r[0] for r in rows]
        hindi_ref = [r[1] for r in rows]
        zs_bleu, zs_chrf, zs_out = results[(label, "zero_shot")]
        ft_bleu, ft_chrf, ft_out = results[(label, "fine_tuned")]
        with open(f"{OUT_DIR}/eval_results_{label}_test.txt", "w", encoding="utf-8") as f:
            f.write(f"Test set: {path} ({len(rows)} examples)\n")
            f.write(f"Zero-shot baseline: BLEU={zs_bleu:.2f} chrF++={zs_chrf:.2f}\n")
            f.write(f"LoRA fine-tuned (real data): BLEU={ft_bleu:.2f} chrF++={ft_chrf:.2f}\n\n")
            f.write("Sample comparisons:\n")
            for i in range(len(prakrit)):
                f.write(f"\n[{i}] Prakrit:    {prakrit[i]}\n")
                f.write(f"    Hindi ref:  {hindi_ref[i]}\n")
                f.write(f"    Zero-shot:  {zs_out[i]}\n")
                f.write(f"    Fine-tuned: {ft_out[i]}\n")
        print(f"Wrote {OUT_DIR}/eval_results_{label}_test.txt", file=sys.stderr)


if __name__ == "__main__":
    main()
