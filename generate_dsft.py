from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, json, os
from tqdm import tqdm

# === Config ===
MODEL_NAME = "gpt2-large"
PREFIX_FILE = "prefixes.jsonl"
OUTPUT_FILE = "sft_dataset.jsonl"
SUFFIX_TOKENS = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_EVERY = 10   # save every N generations

# === Load model/tokenizer ===
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# === Load prefixes ===
prefixes = []
with open(PREFIX_FILE, "r", encoding="utf-8") as f:
    for line in f:
        prefixes.append(json.loads(line)["prefix"])

# === Resume support: count existing lines ===
done = 0
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        done = sum(1 for _ in f)
    print(f"[Resume] Found {done} existing pairs, resuming from there.")

# === Generate suffixes ===
pairs_done = []
for idx in tqdm(range(done, len(prefixes)), desc="Generating suffixes"):
    prefix = prefixes[idx]

    inputs = tok(prefix, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=SUFFIX_TOKENS,
            do_sample=False,  # greedy decode
            pad_token_id=tok.eos_token_id,
        )

    full = tok.decode(outputs[0], skip_special_tokens=True)
    suffix = full[len(prefix):].strip()

    pairs_done.append({"prefix": prefix, "suffix": suffix})

    # Save every SAVE_EVERY
    if (idx + 1) % SAVE_EVERY == 0 or idx == len(prefixes) - 1:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            for p in pairs_done:
                f.write(json.dumps(p) + "\n")
        print(f"[Checkpoint] Saved {len(pairs_done)} new pairs (up to {idx+1}).")
        pairs_done = []

print("✅ Done. All pairs saved to", OUTPUT_FILE)
