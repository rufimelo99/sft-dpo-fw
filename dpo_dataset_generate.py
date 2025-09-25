import os, json, torch, random
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from datasets import load_dataset

# === Config ===
MODEL_NAME = "./investigator_checkpoints/checkpoint-507"  # investigator model
TARGET_MODEL_NAME = "gpt2-large"           # pm for scoring
DATA_FILE = "bad_suffixes.jsonl"            # suffix dataset
OUTPUT_FILE = "dpo_dataset.jsonl"          # where preference pairs are saved
PROGRESS_FILE = "dpo_dataset_generation_progress.json"  # progress checkpoint

NUM_CANDIDATES = 5       # how many candidate prefixes per suffix
PAIRS_PER_SUFFIX = 4     # how many winner/loser pairs to save
MAX_NEW_TOKENS = 40
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Load models ===
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

investigator = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE).eval()
target_model = AutoModelForCausalLM.from_pretrained(TARGET_MODEL_NAME).to(DEVICE).eval()

# === Load suffix dataset ===
dataset = load_dataset("json", data_files=DATA_FILE, split="train")
suffixes = [ex["suffix"] for ex in dataset]

# === Resume support ===
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, "r") as f:
        progress = json.load(f)
    start_idx = progress.get("last_suffix_idx", 0) + 1
else:
    start_idx = 0

print(f"Starting from suffix {start_idx}/{len(suffixes)}")

# === Helper: score pm(y|x) ===
def score_prefix(prefix, suffix):
    text = f"{prefix} {suffix}"
    enc = tok(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = target_model(**enc, labels=enc["input_ids"])
    # average log prob per token
    return -out.loss.item()

# === Main loop ===
with open(OUTPUT_FILE, "a") as fout:
    for i in tqdm(range(start_idx, len(suffixes)), desc="Processing suffixes"):
        suffix = suffixes[i]

        # 1. Generate candidate prefixes from investigator
        prompt = f"<suffix> {suffix} <prefix>"
        inputs = tok(prompt, return_tensors="pt").to(DEVICE)
        candidates = []
        for _ in range(NUM_CANDIDATES):
            out = investigator.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
                pad_token_id=tok.eos_token_id
            )
            decoded = tok.decode(out[0], skip_special_tokens=True)
            prefix = decoded.split("<prefix>")[-1].strip()
            candidates.append(prefix)

        # 2. Score candidates with pm(y|x)
        scored = [(p, score_prefix(p, suffix)) for p in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)

        # 3. Build preference pairs
        pairs = []
        # always include best vs worst
        if len(scored) >= 2:
            pairs.append({"chosen": scored[0][0], "rejected": scored[-1][0], "suffix": suffix})
        # sample more pairs if needed
        while len(pairs) < PAIRS_PER_SUFFIX and len(scored) > 1:
            a, b = random.sample(scored, 2)
            if a[1] > b[1]:
                w, l = a, b
            else:
                w, l = b, a
            pairs.append({"chosen": w[0], "rejected": l[0], "suffix": suffix})

        # 4. Save all pairs for this suffix in one go
        for p in pairs:
            fout.write(json.dumps(p) + "\n")
        fout.flush()

        # 5. Update progress file (suffix-level)
        with open(PROGRESS_FILE, "w") as f:
            json.dump({
                "last_suffix_idx": i,
                "pairs_generated": (i + 1) * PAIRS_PER_SUFFIX
            }, f)

print("✅ DPO dataset generation complete")
