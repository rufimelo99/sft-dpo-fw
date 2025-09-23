from datasets import load_dataset
from transformers import AutoTokenizer
import random, json

# === Config ===
DATASET_NAME = "wikitext"       # later swap to "HuggingFaceFW/fineweb"
DATASET_CONFIG = "wikitext-103-v1"
SPLIT = "train"
NUM_PREFIXES = 1000
OUTPUT_FILE = "prefixes.jsonl"
PREFIX_TOKENS = 64
MODEL_NAME = "gpt2-large"       # tokenizer must match your SFT model

# === Load dataset ===
if DATASET_CONFIG:
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=SPLIT)
else:
    dataset = load_dataset(DATASET_NAME, split=SPLIT)

# === Load tokenizer ===
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None:  # GPT-2 tokenizer has no pad by default
    tok.pad_token = tok.eos_token

# === Sample rows ===
indices = random.sample(range(len(dataset)), NUM_PREFIXES)
prefixes = []

for i in indices:
    text = dataset[i]["text"].strip()
    if not text:
        continue

    # Tokenize
    tokens = tok.encode(text)
    if len(tokens) < PREFIX_TOKENS:
        continue  # skip very short examples

    # Truncate to exactly PREFIX_TOKENS
    truncated = tokens[:PREFIX_TOKENS]
    decoded = tok.decode(truncated)

    prefixes.append(decoded)

# === Save as JSONL ===
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for p in prefixes:
        f.write(json.dumps({"prefix": p}) + "\n")

print(f"Saved {len(prefixes)} prefixes to {OUTPUT_FILE}")
print("Example prefix:\n", prefixes[0])
