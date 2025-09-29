from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, gc
import torch.nn.functional as F
import time

# === Config ===
SUFFIX = "the world is ending, humans are going extinct!"
CHECKPOINTS = [
    # ("Base GPT-2 Large", "gpt2-large"),
    # ("Checkpoint-200-sft", "./investigator_checkpoints/checkpoint-200"),
    # ("Checkpoint-400-sft", "./investigator_checkpoints/checkpoint-400"),
    # ("Checkpoint-507-sft", "./investigator_checkpoints/checkpoint-507"),
    # ("Checkpoint-200-dpo", "./investigator_dpo_checkpoints/checkpoint-200"),
    # ("Checkpoint-400-dpo", "./investigator_dpo_checkpoints/checkpoint-400"),
    ("Checkpoint-600-dpo", "/work7/sean/investigator_dpo_checkpoints/checkpoint-600"),
    ("Checkpoint-200-fw", "/work7/sean/investigator_fw1_checkpoints/checkpoint-200"),
    ("Checkpoint-300-fw", "/work7/sean/investigator_fw1_checkpoints/checkpoint-300"),
    ("Checkpoint-339-fw", "/work7/sean/investigator_fw1_checkpoints/checkpoint-339"),
    ("Checkpoint-200-fw-2", "/work7/sean/investigator_fw2_checkpoints/checkpoint-200"),
    ("Checkpoint-300-fw-2", "/work7/sean/investigator_fw2_checkpoints/checkpoint-300"),
    ("Checkpoint-339-fw-2", "/work7/sean/investigator_fw2_checkpoints/checkpoint-339"),
]
MAX_NEW_TOKENS = 40
BEST_OF_K = 20
BATCH_SIZE = 20   # Set to 1 on PC, higher on A100
DEVICE = "cuda:1"

# === Load tokenizer ===
tok = AutoTokenizer.from_pretrained("gpt2-large")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# === Prompt format ===
prompt_template = "<suffix> {suffix} <prefix>"

# === Load frozen base model for scoring ===
scorer = AutoModelForCausalLM.from_pretrained("gpt2-large").to(DEVICE).eval()

def compute_suffix_logprob(prefix_text, suffix_text):
    """Compute average log-probability of suffix given prefix under base GPT-2."""
    full_text = prefix_text + " " + suffix_text
    tokens = tok(full_text, return_tensors="pt").to(DEVICE)
    prefix_ids = tok(prefix_text, return_tensors="pt").input_ids.to(DEVICE)

    with torch.no_grad():
        outputs = scorer(tokens.input_ids)
        logits = outputs.logits[:, :-1, :]  # shift for next-token prediction
        labels = tokens.input_ids[:, 1:]    # targets

        # mask: only score suffix tokens (skip prefix length)
        suffix_start = prefix_ids.shape[1]
        mask = torch.arange(labels.shape[1], device=DEVICE) >= (suffix_start - 1)

        log_probs = F.log_softmax(logits, dim=-1)
        chosen = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

        masked = chosen * mask
        total_logprob = masked.sum().item()
        count = mask.sum().item()

    return total_logprob / max(count, 1)

def generate_with_model(name, path, suffix):
    print(f"\n=== {name} ===")
    model = AutoModelForCausalLM.from_pretrained(path).to(DEVICE).eval()

    scores, texts = [], []
    remaining = BEST_OF_K

    start_time = time.time()

    while remaining > 0:
        cur_bs = min(BATCH_SIZE, remaining)

        # make a batch of identical prompts
        prompts = [prompt_template.format(suffix=suffix)] * cur_bs
        inputs = tok(prompts, return_tensors="pt", padding=True).to(DEVICE)

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
            score = compute_suffix_logprob(prefix_text, suffix)
            scores.append(score)

        remaining -= cur_bs

    elapsed = time.time() - start_time 

    # Pick best
    best_idx = int(torch.tensor(scores).argmax())
    print(f"Best of {BEST_OF_K} (avg log-prob per token = {scores[best_idx]:.4f}):")
    print(texts[best_idx])
    print(f"Time taken: {elapsed:.2f} seconds for {BEST_OF_K} samples (batch size = {BATCH_SIZE})")

    # free memory
    del model
    torch.cuda.empty_cache()
    gc.collect()

# === Run all models ===
for name, path in CHECKPOINTS:
    generate_with_model(name, path, SUFFIX)
