import os, json, random, gc
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from tqdm import tqdm
from datasets import load_dataset

# === Config ===
INVESTIGATOR_MODEL = "/work7/sean/investigator_dpo_checkpoints/checkpoint-600"  # DPO model
TARGET_MODEL_NAME = "gpt2-large"    # base LM (p_m)
DATA_FILE = "bad_suffixes.jsonl"    # suffix dataset
OUTPUT_FILE = "fw1_dataset.jsonl"   # chosen/rejected pairs
PROGRESS_FILE = "fw1_generation_progress.json"

NUM_CANDIDATES = 5       # candidate prefixes per suffix
PAIRS_PER_SUFFIX = 4     # pairs to save per suffix
MAX_NEW_TOKENS = 40
BATCH_SIZE_SUFFIXES = 16 # suffixes processed in parallel
MIN_PREFIX_LEN = 5 # minimum prefix len we'll keep in dataset
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"DATA_FILE not found: {DATA_FILE}. Check path and filename.")

os.environ["VLLM_LOG_LEVEL"] = "WARNING"

# === Load tokenizer ===
tok = AutoTokenizer.from_pretrained(TARGET_MODEL_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# === Load models ===
llm = LLM(model=INVESTIGATOR_MODEL)  # investigator (prefix generator)
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

# === Correct scoring: log p_m(y|x) (suffix only) ===
def score_prefix(prefix, suffix):
    """Compute avg log p_m(y|x) under base LM, only over suffix tokens."""
    full_text = f"{prefix} {suffix}"
    full_enc = tok(full_text, return_tensors="pt").to(DEVICE)
    prefix_enc = tok(prefix, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = target_model(full_enc.input_ids)
        logits = outputs.logits[:, :-1, :]   # predict next token
        labels = full_enc.input_ids[:, 1:]  # shifted targets

        # mask: only keep suffix tokens
        suffix_start = prefix_enc.input_ids.shape[1]
        mask = torch.arange(labels.shape[1], device=DEVICE) >= (suffix_start - 1)

        log_probs = F.log_softmax(logits, dim=-1)
        chosen = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

        masked = chosen * mask
        total_logprob = masked.sum().item()
        count = mask.sum().item()

    return total_logprob / max(count, 1)

# === Main loop ===
with open(OUTPUT_FILE, "a") as fout:
    for batch_start in tqdm(range(start_idx, len(suffixes), BATCH_SIZE_SUFFIXES),
                            desc="Processing suffixes"):
        batch_suffixes = suffixes[batch_start: batch_start + BATCH_SIZE_SUFFIXES]

        # 1. Generate candidate prefixes for all suffixes with vLLM
        prompts = [f"<suffix> {s} <prefix>" for s in batch_suffixes]
        sampling_params = SamplingParams(
            n=NUM_CANDIDATES,
            max_tokens=MAX_NEW_TOKENS,
            temperature=0.8,
            top_p=0.9
        )
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

        # 2. For each suffix in the batch
        for j, sfx in enumerate(batch_suffixes):
            candidates = []
            for o in outputs[j].outputs:
                decoded = o.text.strip()
                prefix = decoded.split("<prefix>")[-1].strip()
                if len(prefix) >= MIN_PREFIX_LEN:  # drop trivial prefixes
                    candidates.append(prefix)

            # if fewer than 2 candidates, skip this suffix
            if len(candidates) < 2:
                continue

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
