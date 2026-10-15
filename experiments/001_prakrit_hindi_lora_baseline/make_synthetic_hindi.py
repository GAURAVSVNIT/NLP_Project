import csv
import sys
from pathlib import Path

import torch
from transformers import MarianMTModel, MarianTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "models/opus-mt-en-hi"
BATCH_SIZE = 32


def translate_en_to_hi(sentences, tokenizer, model):
    outputs = []
    for i in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[i : i + BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(DEVICE)
        with torch.no_grad():
            generated = model.generate(**inputs, num_beams=4, max_length=256)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        outputs.extend(decoded)
        print(f"  translated {min(i + BATCH_SIZE, len(sentences))}/{len(sentences)}", file=sys.stderr)
    return outputs


def main():
    tokenizer = MarianTokenizer.from_pretrained(MODEL_DIR)
    model = MarianMTModel.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()

    data_dir = Path("data")

    # 1) Train set: final_train_clean.tsv (prakrit \t english) -> add hindi column
    train_rows = []
    with open(data_dir / "final_train_clean.tsv", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            if len(row) >= 2:
                train_rows.append((row[0], row[1]))

    print(f"Loaded {len(train_rows)} train pairs", file=sys.stderr)
    english_train = [r[1] for r in train_rows]
    hindi_train = translate_en_to_hi(english_train, tokenizer, model)

    with open(data_dir / "prakrit_hindi_train.tsv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["prakrit", "hindi", "english_ref"])
        for (prakrit, english), hindi in zip(train_rows, hindi_train):
            writer.writerow([prakrit, hindi, english])

    # 2) Test set: 100_test_pra.txt / 100_test_eng.txt -> prakrit_hindi_test.tsv
    with open(data_dir / "100_test_pra.txt", encoding="utf-8") as f:
        test_prakrit = [line.strip() for line in f if line.strip()]
    with open(data_dir / "100_test_eng.txt", encoding="utf-8") as f:
        test_english = [line.strip() for line in f if line.strip()]

    assert len(test_prakrit) == len(test_english), "test set line count mismatch"
    print(f"Loaded {len(test_prakrit)} test pairs", file=sys.stderr)
    hindi_test = translate_en_to_hi(test_english, tokenizer, model)

    with open(data_dir / "prakrit_hindi_test.tsv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["prakrit", "hindi", "english_ref"])
        for prakrit, hindi, english in zip(test_prakrit, hindi_test, test_english):
            writer.writerow([prakrit, hindi, english])

    print("Done. Wrote data/prakrit_hindi_train.tsv and data/prakrit_hindi_test.tsv", file=sys.stderr)


if __name__ == "__main__":
    main()
