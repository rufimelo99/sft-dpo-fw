from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json, os
from tqdm import tqdm
import sys

# === Config ===
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"   # or any other HF model
PREFIX_FILE = "prefixes_fineweb.jsonl"
OUTPUT_FILE = "sft_dataset.jsonl"
SUFFIX_TOKENS = 64
BATCH_SIZE = 32        # adjust depending on GPU memory (A100 can handle 32–64 easily)
SAVE_EVERY = 500       # save every N generations

# === Initialize vLLM model ===
llm = LLM(MODEL_NAME, dtype="bfloat16")   # or "float16" if you prefer
sampling_params = SamplingParams(
    max_tokens=SUFFIX_TOKENS,
    temperature=0.0,   # greedy decode
    top_p=1.0,
)

# === Verify prefix file exists ===
if not os.path.exists(PREFIX_FILE):
    sys.exit(f"Prefix file not found: {PREFIX_FILE}")

# === Load prefixes ===
prefixes = []
with open(PREFIX_FILE, "r", encoding="utf-8") as f:
    for line in f:
        prefixes.append(json.loads(line)["prefix"])

# === Resume support ===
done = 0
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        done = sum(1 for _ in f)
    print(f"[Resume] Found {done} existing pairs, resuming from there.")

# === Generation loop (batched) ===
pairs_done = []
for start in tqdm(range(done, len(prefixes), BATCH_SIZE), desc="Generating suffixes"):
    batch = prefixes[start:start+BATCH_SIZE]

    # Run inference with vLLM
    outputs = llm.generate(batch, sampling_params, use_tqdm=False)

    for prefix, out in zip(batch, outputs):
        suffix = out.outputs[0].text.strip()

        pairs_done.append({"prefix": prefix, "suffix": suffix})

    # Save periodically
    if len(pairs_done) >= SAVE_EVERY or start + BATCH_SIZE >= len(prefixes):
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            for p in pairs_done:
                f.write(json.dumps(p) + "\n")
        print(f"[Checkpoint] Saved {len(pairs_done)} new pairs (total {start+len(batch)}).")
        pairs_done = []

print("✅ Done. All pairs saved to", OUTPUT_FILE)
