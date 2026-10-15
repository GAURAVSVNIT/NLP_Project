import csv
import sys
from pathlib import Path

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
OUTPUT_DIR = "models/prakrit-hindi-lora-domain-mix"
LANG_TAG = "hin_Deva"  # script-compatible routing: Prakrit source is tagged as Hindi
MAX_LEN = 160
VAL_SIZE = 40


def load_tsv(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append((row["prakrit"], row["hindi"]))
    return rows


def main():
    ip = IndicProcessor(inference=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_rows = load_tsv("data/prakrit_hindi_domain_mix_train.tsv")

    # small held-out validation slice carved out of train for eval during training
    val_rows = train_rows[-VAL_SIZE:]
    train_rows = train_rows[:-VAL_SIZE]

    print(f"train={len(train_rows)} val={len(val_rows)}", file=sys.stderr)

    def to_dataset(rows):
        prakrit = [r[0] for r in rows]
        hindi = [r[1] for r in rows]
        # source gets the "src_lang tgt_lang" tag prefix the encoder expects;
        # target must NOT get that prefix (is_target=True), or the model learns
        # to predict the literal tag tokens instead of the translation.
        src = ip.preprocess_batch(prakrit, src_lang=LANG_TAG, tgt_lang=LANG_TAG)
        tgt = ip.preprocess_batch(hindi, src_lang=LANG_TAG, is_target=True)
        return Dataset.from_dict({"source": src, "target": tgt})

    train_ds = to_dataset(train_rows)
    val_ds = to_dataset(val_rows)

    def preprocess(batch):
        model_inputs = tokenizer(batch["source"], truncation=True, max_length=MAX_LEN)
        labels = tokenizer(text_target=batch["target"], truncation=True, max_length=MAX_LEN)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_tok = train_ds.map(preprocess, batched=True, remove_columns=["source", "target"])
    val_tok = val_ds.map(preprocess, batched=True, remove_columns=["source", "target"])

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=3e-4,
        num_train_epochs=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=MAX_LEN,
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=True,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=collator,
        processing_class=tokenizer,
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved LoRA adapter to {OUTPUT_DIR}", file=sys.stderr)


if __name__ == "__main__":
    main()
