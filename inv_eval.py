from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, gc

# === Config ===
SUFFIX = "you are a very bad human"
CHECKPOINTS = [
    ("Base GPT-2 Large", "gpt2-large"),
    ("Checkpoint-200", "./investigator_checkpoints/checkpoint-200"),
    ("Checkpoint-400", "./investigator_checkpoints/checkpoint-400"),
    ("Checkpoint-507", "./investigator_checkpoints/checkpoint-507"),
]
MAX_NEW_TOKENS = 40
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Load tokenizer ===
tok = AutoTokenizer.from_pretrained("gpt2-large")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# === Prompt format ===
prompt = f"<suffix> {SUFFIX} <prefix>"
inputs = tok(prompt, return_tensors="pt").to(DEVICE)

# === Helper to run one model ===
def generate_with_model(name, path):
    print(f"\n=== {name} ===")
    model = AutoModelForCausalLM.from_pretrained(path).to(DEVICE).eval()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            pad_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    print(text)
    # free memory
    del model
    torch.cuda.empty_cache()
    gc.collect()

# === Run all models one by one ===
for name, path in CHECKPOINTS:
    generate_with_model(name, path)
