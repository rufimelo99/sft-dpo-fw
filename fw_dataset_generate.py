import os, json, random, gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from tqdm import tqdm
from datasets import load_dataset

# === Config ===
INVESTIGATOR_MODEL = "/work7/sean/investigator_dpo_checkpoints/checkpoint-600"  # DPO model
TARGET_MODEL_NAME = "gpt2-large"    # base LM pm
DATA_FILE = "bad_suffixes.jsonl"    # suffix dataset
OUTPUT_FILE = "fw1_dataset.jsonl"   # chosen/rejected pairs
PROGRESS_FILE = "fw1_generation_progress.json"

NUM_CANDIDATES = 5       # candidate prefixes per suffix
PAIRS_PER_SUFFIX = 4     # pairs to save per suffix
MAX_NEW_TOKENS = 40
BATCH_SIZE_SUFFIXES = 16 # how many suffixes to process in parallel
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"DATA_FILE not found: {DATA_FILE}. Check path and filename.")

os.environ["VLLM_LOG_LEVEL"] = "WARNING" # setting vllm to warning so there is not so many progress logs

# === Load tokenizer ===
tok = AutoTokenizer.from_pretrained(TARGET_MODEL_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# === Load models ===
# Investigator (prefix generator)
llm = LLM(model=INVESTIGATOR_MODEL)
# Base LM for scoring
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

# === Scoring helper ===
def score_prefix(prefix, suffix):
    """Compute avg log p_m(y|x) under base LM."""
    text = f"{prefix} {suffix}"
    enc = tok(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = target_model(**enc, labels=enc["input_ids"])
    return -out.loss.item()

# === Main loop ===
with open(OUTPUT_FILE, "a") as fout:
    for batch_start in tqdm(range(start_idx, len(suffixes), BATCH_SIZE_SUFFIXES),
                            desc="Processing suffixes"):
        batch_suffixes = suffixes[batch_start: batch_start + BATCH_SIZE_SUFFIXES]

        # 1. Generate candidate prefixes in parallel with vLLM
        prompts = [f"<suffix> {s} <prefix>" for s in batch_suffixes]
        sampling_params = SamplingParams(
            n=NUM_CANDIDATES,
            max_tokens=MAX_NEW_TOKENS,
            temperature=0.8,
            top_p=0.9
        )
        outputs = llm.generate(prompts, sampling_params)

        # 2. For each suffix in the batch
        for j, sfx in enumerate(batch_suffixes):
            candidates = []
            for o in outputs[j].outputs:
                decoded = o.text.strip()
                prefix = decoded.split("<prefix>")[-1].strip()
                candidates.append(prefix)

            # 3. Score candidates with base LM
            scored = [(p, score_prefix(p, sfx)) for p in candidates]
            scored.sort(key=lambda x: x[1], reverse=True)

            # 4. Build preference pairs
            pairs = []
            if len(scored) >= 2:
                # always include best vs worst
                pairs.append({"chosen": scored[0][0], "rejected": scored[-1][0], "suffix": sfx})
            while len(pairs) < PAIRS_PER_SUFFIX and len(scored) > 1:
                a, b = random.sample(scored, 2)
                if a[1] > b[1]:
                    w, l = a, b
                else:
                    w, l = b, a
                pairs.append({"chosen": w[0], "rejected": l[0], "suffix": sfx})

            # 5. Write pairs
            for p in pairs:
                fout.write(json.dumps(p) + "\n")
            fout.flush()

            # 6. Update progress (suffix-level)
            with open(PROGRESS_FILE, "w") as f:
                json.dump({
                    "last_suffix_idx": batch_start + j,
                    "pairs_generated": (batch_start + j + 1) * PAIRS_PER_SUFFIX
                }, f)

        torch.cuda.empty_cache()
        gc.collect()

print("✅ FW-1 dataset generation complete")
