import os, json, random, gc
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from tqdm import tqdm
from datasets import load_dataset

# === Config ===
INVESTIGATOR_MODEL = "/work7/sean/l8b_investigator_sft_checkpoints/checkpoint-846"  # FW1 model
TARGET_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"    # base LM (p_m)
PENALIZE_MODEL = ""      # model to penalize (FW1). "" = no penalty
LAMBDA = 1.0  # weight for penalty term

DATA_FILE = "fineweb_train.jsonl"    # suffix dataset
OUTPUT_FILE = "fineweb_dpo_l8b.jsonl"   # chosen/rejected pairs
PROGRESS_FILE = "l8b_dpo_progress.json"

NUM_CANDIDATES = 4       # candidate prefixes per suffix
PAIRS_PER_SUFFIX = 2     # pairs to save per suffix
MAX_NEW_TOKENS = 64
BATCH_SIZE_SUFFIXES = 8 # suffixes processed in parallel
MIN_PREFIX_LEN = 10 # minimum prefix len we'll keep in dataset
MINIMUM_SCORE_DIFF = 0.05 # minimum score diff between pairs that we'll accept
DEVICE_INV = "cuda:0" # device for investigator
DEVICE_TARGET = "cuda:1" # device for target model, for scoring proposed prefixes
DEVICE_PENALIZE = "" # device for the previous fw model, for penalizing repeated proposals

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"DATA_FILE not found: {DATA_FILE}. Check path and filename.")

# === Load tokenizer ===
tok = AutoTokenizer.from_pretrained(TARGET_MODEL_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# === Load models ===
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5" # controls what gpu the investigator uses
llm = LLM(model=INVESTIGATOR_MODEL, dtype="bfloat16", tensor_parallel_size=1)  # investigator (prefix generator)
target_model = AutoModelForCausalLM.from_pretrained(TARGET_MODEL_NAME, torch_dtype=torch.bfloat16).to(DEVICE_TARGET).eval()

penalize_model = None
if PENALIZE_MODEL:
    penalize_model = AutoModelForCausalLM.from_pretrained(PENALIZE_MODEL).to(DEVICE_PENALIZE).eval()

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

# === Scoring functions ===
def score_suffix_given_prefix(prefix, suffix):
    """Compute avg log p_m(y|x) under base LM, only over suffix tokens."""
    full_text = f"{prefix} {suffix}"
    full_enc = tok(full_text, return_tensors="pt").to(DEVICE_TARGET)
    prefix_enc = tok(prefix, return_tensors="pt").to(DEVICE_TARGET)

    with torch.no_grad():
        outputs = target_model(full_enc.input_ids)
        logits = outputs.logits[:, :-1, :]   # predict next token
        labels = full_enc.input_ids[:, 1:]  # shifted targets

        # mask: only keep suffix tokens
        suffix_start = prefix_enc.input_ids.shape[1]
        mask = torch.arange(labels.shape[1], device=DEVICE_TARGET) >= (suffix_start - 1)

        log_probs = F.log_softmax(logits, dim=-1)
        chosen = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

        masked = chosen * mask
        total_logprob = masked.sum().item()
        count = mask.sum().item()

    return total_logprob / max(count, 1)

def score_prefix_given_suffix(prefix, suffix):
    """Compute avg log p_prev(x|y) under penalize model, only over prefix tokens."""
    if not penalize_model:
        return 0.0  # no penalty if model not provided

    full_text = f"<suffix> {suffix} <prefix> {prefix}"
    full_enc = tok(full_text, return_tensors="pt").to(DEVICE_PENALIZE)
    suffix_enc = tok(f"<suffix> {suffix} <prefix>", return_tensors="pt").to(DEVICE_PENALIZE)

    with torch.no_grad():
        outputs = penalize_model(full_enc.input_ids)
        logits = outputs.logits[:, :-1, :]
        labels = full_enc.input_ids[:, 1:]

        # mask: only keep prefix tokens
        prefix_start = suffix_enc.input_ids.shape[1]
        mask = torch.arange(labels.shape[1], device=DEVICE_PENALIZE) >= (prefix_start - 1)

        log_probs = F.log_softmax(logits, dim=-1)
        chosen = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

        masked = chosen * mask
        total_logprob = masked.sum().item()
        count = mask.sum().item()

    return total_logprob / max(count, 1)

def score_total(prefix, suffix):
    """Combined reward: log p_m(y|x) - λ log p_prev(x|y)."""
    base_score = score_suffix_given_prefix(prefix, suffix)
    penalty = score_prefix_given_suffix(prefix, suffix)
    return base_score - LAMBDA * penalty

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
            temperature=1.1,
            top_p=0.95
        )
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

        # 2. For each suffix in the batch
        for j, sfx in enumerate(batch_suffixes):
            candidates = set()
            for o in outputs[j].outputs:
                decoded = o.text.strip()
                prefix = decoded.split("<prefix>")[-1].strip()
                if len(prefix) >= MIN_PREFIX_LEN:
                    candidates.add(prefix)

            # if fewer than 2 candidates, skip this suffix
            if len(candidates) < 2:
                continue

            # 3. Score candidates
            scored = [(p, score_total(p, sfx)) for p in candidates]
            scored.sort(key=lambda x: x[1], reverse=True)

            # 4. Build preference pairs
            pairs = []
            if len(scored) >= 2 and abs(scored[0][1] - scored[-1][1]) >= MINIMUM_SCORE_DIFF:
                # always include best vs worst
                pairs.append({"chosen": scored[0][0], "rejected": scored[-1][0], "suffix": sfx})
            else:
                continue

            # 5. Write pairs
            for p in pairs:
                fout.write(json.dumps(p) + "\n")
            fout.flush()

            # 6. Update progress
            with open(PROGRESS_FILE, "w") as f:
                json.dump({
                    "last_suffix_idx": batch_start + j,
                }, f)

        torch.cuda.empty_cache()
        gc.collect()

print("✅ FW/DPO dataset generation complete")
