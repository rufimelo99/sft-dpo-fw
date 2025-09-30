from datasets import load_dataset
from transformers import AutoTokenizer
import random, json

# === Config ===
DATASET_NAME = "HuggingFaceFW/fineweb"   # FineWeb dataset
SPLIT = "train"                          # single split, streamed
NUM_PREFIXES = 5000
OUTPUT_FILE = "prefixes_fineweb.jsonl"
PREFIX_TOKENS = 64
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"   # tokenizer must match model

# === Load dataset in streaming mode (no full download) ===
dataset = load_dataset(DATASET_NAME, split=SPLIT, streaming=True)

# === Load tokenizer ===
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# === Sample prefixes ===
prefixes = []
for example in dataset:
    if len(prefixes) >= NUM_PREFIXES:
        break

    text = example["text"].strip()
    if not text:
        continue

    tokens = tok.encode(text)
    if len(tokens) < PREFIX_TOKENS:
        continue

    truncated = tokens[:PREFIX_TOKENS]
    decoded = tok.decode(truncated, skip_special_tokens=True)
    prefixes.append(decoded)

# === Save ===
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for p in prefixes:
        f.write(json.dumps({"prefix": p}) + "\n")

print(f"Saved {len(prefixes)} prefixes to {OUTPUT_FILE}")
print("Example prefix:\n", prefixes[0])
