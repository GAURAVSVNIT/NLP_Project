"""
Experiment 008: 5-fold cross-validation over the grammar drills, comparing
  - "drill":  train on drill folds only (Exp 007 config)
  - "mix":    train on drill folds + Agama real data (Exp 005 config)
Each drill pair is held out exactly once, so the pooled score covers all
267 drill pairs instead of Exp 005/007's 20-example test.

Hyperparameters, LoRA config and the 40-row validation slice match
Exp 005/007. Fold assignment is deterministic (seed 123).

Usage (from project root):
    python experiments/008_drill_kfold_cv/run_drill_kfold_cv.py
Results are appended to experiments/008_drill_kfold_cv/cv_results.jsonl and
resumable: finished (config, fold) pairs are skipped on rerun.
"""
import csv
import json
import random
import sys
from pathlib import Path

import evaluate
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from IndicTransToolkit.processor import IndicProcessor

MODEL_PATH = "models/indictrans2-indic-indic-1B"
LANG_TAG = "hin_Deva"
MAX_LEN = 160
VAL_SIZE = 40
N_FOLDS = 5
DATA = Path("data")
WORK = Path("models/cv_008")
RESULTS = Path("experiments/008_drill_kfold_cv/cv_results.jsonl")


def load_pairs(path, prakrit_col="prakrit", hindi_col="hindi"):
    with open(path, encoding="utf-8") as f:
        return [(r[prakrit_col], r[hindi_col]) for r in csv.DictReader(f, delimiter="\t")]


def make_folds(drills):
    rng = random.Random(123)
    idx = list(range(len(drills)))
    rng.shuffle(idx)
    size = (len(idx) + N_FOLDS - 1) // N_FOLDS
    return [[drills[i] for i in idx[k * size:(k + 1) * size]] for k in range(N_FOLDS)]


def split_for(config, fold_test, drills, agama_rows, agama_test_prakrit):
    """Return (fit_rows, val_rows) for a config.

    drill / mix: validation = last 40 rows of the shuffled training pool
                 (drill rows for drill, mostly Agama rows for mix).
    mixdv:       mix training data, but validation = 40 drill rows, so
                 checkpoint selection is on the drill domain (fixes the
                 selection confound in the mix config).
    """
    held = {p for p, _ in fold_test}
    drill_rows = [r for r in drills if r[0] not in held]
    if config == "mixdv":
        random.Random(123).shuffle(drill_rows)
        val_rows = drill_rows[-VAL_SIZE:]
        fit_rows = drill_rows[:-VAL_SIZE]
        seen = set(agama_test_prakrit) | {p for p, _ in drill_rows}
        for pra, hi in agama_rows:
            if pra not in seen:
                seen.add(pra)
                fit_rows.append((pra, hi))
        random.Random(123).shuffle(fit_rows)
        return fit_rows, val_rows

    rows = list(drill_rows)
    if config == "mix":
        seen = set(agama_test_prakrit) | {p for p, _ in rows}
        for pra, hi in agama_rows:
            if pra not in seen:
                seen.add(pra)
                rows.append((pra, hi))
    random.Random(123).shuffle(rows)
    return rows[:-VAL_SIZE], rows[-VAL_SIZE:]


def train_and_translate(config, fold_idx, fit_rows, val_rows, test_rows, ip, ip_inf, tokenizer):
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], task_type="SEQ_2_SEQ_LM",
    ))

    def to_dataset(rows):
        pra = [r[0] for r in rows]
        hi = [r[1] for r in rows]
        src = ip.preprocess_batch(pra, src_lang=LANG_TAG, tgt_lang=LANG_TAG)
        tgt = ip.preprocess_batch(hi, src_lang=LANG_TAG, is_target=True)
        return Dataset.from_dict({"source": src, "target": tgt})

    def preprocess(batch):
        enc = tokenizer(batch["source"], truncation=True, max_length=MAX_LEN)
        enc["labels"] = tokenizer(text_target=batch["target"], truncation=True, max_length=MAX_LEN)["input_ids"]
        return enc

    train_tok = to_dataset(fit_rows).map(preprocess, batched=True, remove_columns=["source", "target"])
    val_tok = to_dataset(val_rows).map(preprocess, batched=True, remove_columns=["source", "target"])

    out_dir = WORK / f"{config}_fold{fold_idx}"
    args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=4, per_device_eval_batch_size=4,
        gradient_accumulation_steps=4, learning_rate=3e-4, num_train_epochs=8,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        logging_steps=20, load_best_model_at_end=True, metric_for_best_model="eval_loss",
        bf16=True, report_to=[],
    )
    trainer = Seq2SeqTrainer(
        model=model, args=args, train_dataset=train_tok, eval_dataset=val_tok,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100),
        processing_class=tokenizer,
    )
    trainer.train()
    model.eval()

    hyps = []
    srcs = [p for p, _ in test_rows]
    for i in range(0, len(srcs), 8):
        batch = srcs[i:i + 8]
        pre = ip_inf.preprocess_batch(batch, src_lang=LANG_TAG, tgt_lang=LANG_TAG)
        inputs = tokenizer(pre, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, num_beams=5, max_length=MAX_LEN,
                                 repetition_penalty=1.3, no_repeat_ngram_size=3)
        hyps.extend(ip_inf.postprocess_batch(tokenizer.batch_decode(gen, skip_special_tokens=True), lang=LANG_TAG))
    del model, trainer
    torch.cuda.empty_cache()
    return hyps


def main():
    drills = load_pairs(DATA / "prakrit_hindi_grammar_drills.tsv")
    agama_rows = load_pairs(DATA / "prakrit_hindi_real_agama.tsv") + load_pairs(DATA / "prakrit_hindi_acharanga_bhashyam.tsv")
    agama_test_prakrit = [p for p, _ in load_pairs(DATA / "prakrit_hindi_real_test.tsv")]
    folds = make_folds(drills)
    assert sum(len(f) for f in folds) == len(drills)

    done = set()
    if RESULTS.exists():
        for line in RESULTS.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["config"], r["fold"]))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    ip = IndicProcessor(inference=False)
    ip_inf = IndicProcessor(inference=True)

    # Optional "config fold" argv runs a single (config, fold) in this process.
    # Each fold runs in its own process so a native crash on one fold cannot
    # take down the rest (the first full run died with exit 139 after fold 1).
    only = (sys.argv[1], int(sys.argv[2])) if len(sys.argv) == 3 else None

    for config in ["drill", "mix", "mixdv"]:
        for k, test_rows in enumerate(folds):
            if only and (config, k) != only:
                continue
            if (config, k) in done:
                print(f"skip {config} fold {k} (done)", file=sys.stderr)
                continue
            fit_rows, val_rows = split_for(config, test_rows, drills, agama_rows, agama_test_prakrit)
            print(f"{config} fold {k}: fit={len(fit_rows)} val={len(val_rows)} test={len(test_rows)}", file=sys.stderr)
            hyps = train_and_translate(config, k, fit_rows, val_rows, test_rows, ip, ip_inf, tokenizer)
            with RESULTS.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "config": config, "fold": k,
                    "prakrit": [p for p, _ in test_rows],
                    "ref": [h for _, h in test_rows],
                    "hyp": hyps,
                }, ensure_ascii=False) + "\n")

    if only:
        return
    # pooled scores per config over all folds
    bleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")
    by_cfg = {}
    for line in RESULTS.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        acc = by_cfg.setdefault(r["config"], {"hyp": [], "ref": []})
        acc["hyp"] += r["hyp"]
        acc["ref"] += r["ref"]
    for cfg, acc in by_cfg.items():
        b = bleu.compute(predictions=acc["hyp"], references=[[x] for x in acc["ref"]])["score"]
        c = chrf.compute(predictions=acc["hyp"], references=[[x] for x in acc["ref"]], word_order=2)["score"]
        print(f"POOLED {cfg}: n={len(acc['ref'])} BLEU={b:.2f} chrF++={c:.2f}", file=sys.stderr)


if __name__ == "__main__":
    main()
