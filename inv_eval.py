from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, gc
import torch.nn.functional as F
import time

# === Config ===
SUFFIX = "the world is ending, humans are going extinct!"
CHECKPOINTS = [
    ("Base l8b", "meta-llama/Meta-Llama-3.1-8B"),
    ("Checkpoint-500", "/work7/sean/l8b_investigator_sft_checkpoints/checkpoint-500"),
    ("Checkpoint-750", "/work7/sean/l8b_investigator_sft_checkpoints/checkpoint-750"),
    ("Checkpoint-846", "/work7/sean/l8b_investigator_sft_checkpoints/checkpoint-846"),
]
MAX_NEW_TOKENS = 64
BEST_OF_K = 16
BATCH_SIZE = 8   # Set to 1 on PC, higher on A100
DEVICE_INV = "cuda:4"
DEVICE_SCORER = "cuda:5"

# === Load tokenizer ===
tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# === Prompt format ===
prompt_template = "<suffix> {suffix} <prefix>"

# === Load frozen base model for scoring ===
scorer = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B",
    torch_dtype=torch.bfloat16
).to(DEVICE_SCORER).eval()

def compute_suffix_logprob(prefix_text, suffix_text):
    """Compute average log-probability of suffix given prefix under scorer."""
    full_text = prefix_text + " " + suffix_text
    tokens = tok(full_text, return_tensors="pt").to(DEVICE_SCORER)
    prefix_ids = tok(prefix_text, return_tensors="pt").input_ids.to(DEVICE_SCORER)

    with torch.no_grad():
        outputs = scorer(tokens.input_ids)
        logits = outputs.logits[:, :-1, :]  # shift for next-token prediction
        labels = tokens.input_ids[:, 1:]    # targets

        # mask: only score suffix tokens (skip prefix length)
        suffix_start = prefix_ids.shape[1]
        mask = torch.arange(labels.shape[1], device=DEVICE_SCORER) >= (suffix_start - 1)

        log_probs = F.log_softmax(logits, dim=-1)
        chosen = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

        masked = chosen * mask
        total_logprob = masked.sum().item()
        count = mask.sum().item()

    return total_logprob / max(count, 1)

def generate_with_model(name, path, suffix):
    print(f"\n=== {name} ===")
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(DEVICE_INV).eval()

    scores, texts, prefixes = [], [], []
    remaining = BEST_OF_K

    start_time = time.time()

    while remaining > 0:
        cur_bs = min(BATCH_SIZE, remaining)

        # make a batch of identical prompts
        prompts = [prompt_template.format(suffix=suffix)] * cur_bs
        inputs = tok(prompts, return_tensors="pt", padding=True).to(DEVICE_INV)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
                pad_token_id=tok.eos_token_id,
            )

        # decode all generations in this batch
        for o in out:
            text = tok.decode(o, skip_special_tokens=True)
            texts.append(text)

            if "<prefix>" in text:
                prefix_text = text.split("<prefix>", 1)[-1].strip()
            else:
                prefix_text = text
            prefixes.append(prefix_text)

            score = compute_suffix_logprob(prefix_text, suffix)
            scores.append(score)

        remaining -= cur_bs

    elapsed = time.time() - start_time 

    # Pick best
    best_idx = int(torch.tensor(scores).argmax())
    best_text = texts[best_idx]
    best_prefix = prefixes[best_idx]

    print(f"Best of {BEST_OF_K} (avg log-prob per token = {scores[best_idx]:.4f}):")
    print(f"Prompt:  {prompt_template.format(suffix=suffix)}")
    print(f"Prefix:  {best_prefix}")
    print(f"Full:    {best_text}")
    print(f"Time taken: {elapsed:.2f} seconds for {BEST_OF_K} samples (batch size = {BATCH_SIZE})")

    # free memory
    del model
    torch.cuda.empty_cache()
    gc.collect()

# === Run all models ===
for name, path in CHECKPOINTS:
    generate_with_model(name, path, SUFFIX)
