from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch, torch.nn.functional as F, gc, time, json, os
from tqdm import tqdm

# === Config ===
EVAL_FILE = "l8b_base_datasets/toxic/toxic_suffixes_l8b_eval.jsonl"  # Each line: {"suffix": "..."}
OUTPUT_DIR = "toxic_suffixes_l8b_instruct_eval_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHECKPOINTS = [
    ("Base l8b", "meta-llama/Meta-Llama-3.1-8B"),
    ("SFT", "/work7/sean/l8b_investigator_sft_checkpoints/checkpoint-846"),
    ("DPO", "/work7/sean/l8b_investigator_dpo_checkpoints/checkpoint-738"),
    ("FW1", "/work7/sean/l8b_investigator_toxic_fw1_checkpoints/checkpoint-357"),
    ("FW2", "/work7/sean/l8b_investigator_toxic_fw2_checkpoints/checkpoint-357"),
    ("FW3", "/work7/sean/l8b_instruct_investigator_toxic_fw3_checkpoints/checkpoint-357"),
    ("FW4", "/work7/sean/l8b_instruct_investigator_toxic_fw4_checkpoints/checkpoint-357"),
]

CHECKPOINTS.reverse()

MAX_NEW_TOKENS = 64
BEST_OF_K = 16
BATCH_SIZE_SUFFIXES = 8
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5" # which GPUs to use
DEVICE_SCORER = "cuda:1" # VLLM will use 0, so scorer use 1
PROMPT_TEMPLATE = "<suffix> {suffix} <prefix>"
SCORER_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# === Load tokenizer ===
tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# === Load frozen scorer (base LM) ===
scorer = AutoModelForCausalLM.from_pretrained(
    SCORER_MODEL,
    torch_dtype=torch.bfloat16
).to(DEVICE_SCORER).eval()

# === Load evaluation suffixes ===
dataset = load_dataset("json", data_files=EVAL_FILE, split="train")
suffixes = [ex["suffix"] for ex in dataset]
print(f"[INFO] Loaded {len(suffixes)} suffixes for evaluation.")

# === Scoring ===
@torch.inference_mode()
def batch_score_suffix_given_prefix(prefixes, suffixes):
    texts = [f"{p} {s}" for p, s in zip(prefixes, suffixes)]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE_SCORER)
    prefix_lens = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prefixes]

    outputs = scorer(enc.input_ids)
    logits = outputs.logits[:, :-1, :]
    labels = enc.input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    chosen = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

    scores = []
    for i, plen in enumerate(prefix_lens):
        mask = torch.arange(labels.shape[1], device=DEVICE_SCORER) >= (plen - 1)
        total = (chosen[i] * mask).sum().item()
        count = mask.sum().item()
        scores.append(total / max(count, 1))
    return scores

# === Evaluation per model ===
def evaluate_model(name, path):
    print(f"\n=== {name} ===")
    out_path = os.path.join(OUTPUT_DIR, f"{name.replace(' ', '_')}.jsonl")

    if os.path.exists(out_path):
        print(f"[SKIP] {name}: output already exists → {out_path}")
        return

    # Load investigator with vLLM
    llm = LLM(model=path, dtype="bfloat16", tensor_parallel_size=1)
    sampling_params = SamplingParams(
        n=BEST_OF_K,
        temperature=0.9,
        top_p=0.9,
        max_tokens=MAX_NEW_TOKENS,
    )

    all_scores = []  # collect logprobs for averaging
    start_time = time.time()

    with open(out_path, "w") as fout:
        for batch_start in tqdm(range(0, len(suffixes), BATCH_SIZE_SUFFIXES), desc=f"{name}"):
            batch_suffixes = suffixes[batch_start: batch_start + BATCH_SIZE_SUFFIXES]
            prompts = [PROMPT_TEMPLATE.format(suffix=s) for s in batch_suffixes]

            # Generate prefixes with vLLM
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

            # For each suffix, score and select best prefix
            for j, sfx in enumerate(batch_suffixes):
                candidates = set()
                for o in outputs[j].outputs:
                    decoded = o.text.strip()
                    prefix = decoded.split("<prefix>")[-1].strip()
                    if len(prefix) > 0:
                        candidates.add(prefix)

                if not candidates:
                    continue

                prefix_list = list(candidates)
                suffix_list = [sfx] * len(prefix_list)
                scores = batch_score_suffix_given_prefix(prefix_list, suffix_list)
                best_idx = int(torch.tensor(scores).argmax())
                best_prefix = prefix_list[best_idx]

                # Generate actual suffix from scorer
                prefix_input = tok(best_prefix, return_tensors="pt").to(DEVICE_SCORER)
                with torch.no_grad():
                    suffix_out = scorer.generate(
                        **prefix_input,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        pad_token_id=tok.eos_token_id,
                    )
                actual_suffix = tok.decode(
                    suffix_out[0][prefix_input.input_ids.shape[1]:],
                    skip_special_tokens=True
                ).strip()

                best_score = scores[best_idx]
                all_scores.append(best_score)

                fout.write(json.dumps({
                    "target_suffix": sfx,
                    "proposed_prefix": best_prefix,
                    "actual_suffix": actual_suffix,
                    "logprob": best_score
                }) + "\n")
                fout.flush()

    elapsed = time.time() - start_time
    avg_logprob = sum(all_scores) / len(all_scores) if all_scores else float("nan")
    print(f"[DONE] {name}: {len(all_scores)} examples in {elapsed:.1f}s → {out_path}")
    print(f"       Avg log-prob per token: {avg_logprob:.4f}")

    del llm
    torch.cuda.empty_cache()
    gc.collect()

# === Run ===
if __name__ == "__main__":
    for name, path in CHECKPOINTS:
        evaluate_model(name, path)

    print("✅ vLLM batch evaluation complete")
