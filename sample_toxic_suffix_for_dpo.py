import json, random
from datasets import load_dataset
from transformers import AutoTokenizer

# === Config ===
DATASET = "allenai/real-toxicity-prompts"
OUTPUT_FILE = "bad_suffixes.jsonl"
NUM_SUFFIXES = 500        # how many to sample
SEED = 42
SAVE_GOLDEN_PREFIX = True
TOXICITY_THRESHOLD = 0.5  # filter: keep only suffixes with toxicity > this
MAX_TOKENS = 64           # cap suffix length
MIN_TOKENS = 5            # optional floor

random.seed(SEED)
tok = AutoTokenizer.from_pretrained("gpt2-large")

# === Load dataset ===
print(f"Loading dataset {DATASET}...")
ds = load_dataset(DATASET, split="train")

# === Collect suffixes (continuations) ===
suffixes = []
for ex in ds:
    suffix_text = ex["continuation"]["text"].strip()

    # some entries don't have a toxicity score → default to 0.0
    tox_score = ex["continuation"].get("toxicity", 0.0) or 0.0  

    if not suffix_text or tox_score < TOXICITY_THRESHOLD:
        continue

    # tokenize to control length
    tokens = tok(suffix_text, truncation=True, max_length=MAX_TOKENS)["input_ids"]
    if len(tokens) < MIN_TOKENS:
        continue

    # decode back to enforce truncation at token level
    suffix_trimmed = tok.decode(tokens, skip_special_tokens=True).strip()

    entry = {"suffix": suffix_trimmed}
    if SAVE_GOLDEN_PREFIX:
        entry["golden_prefix"] = ex["prompt"]["text"].strip()

    suffixes.append(entry)

print(f"Collected {len(suffixes)} candidate suffixes after toxicity & length filtering.")

# === Sample ===
if len(suffixes) > NUM_SUFFIXES:
    sampled = random.sample(suffixes, NUM_SUFFIXES)
else:
    sampled = suffixes

# === Save ===
with open(OUTPUT_FILE, "w") as f:
    for s in sampled:
        f.write(json.dumps(s) + "\n")

print(f"✅ Saved {len(sampled)} bad suffixes to {OUTPUT_FILE}")
