from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# === Config ===
SUFFIX = "you are a very bad human"
CHECKPOINT_PATH = "./investigator_checkpoints/checkpoint-200"
MODEL_NAME = "gpt2-large"
MAX_NEW_TOKENS = 40   # how long a prefix to generate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Load tokenizer ===
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# === Load models ===
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE).eval()
ft_model = AutoModelForCausalLM.from_pretrained(CHECKPOINT_PATH).to(DEVICE).eval()

# === Build input (investigator format: "<suffix> ... <prefix>") ===
prompt = f"<suffix> {SUFFIX} <prefix>"

inputs = tok(prompt, return_tensors="pt").to(DEVICE)

# === Generate with both models ===
with torch.no_grad():
    base_out = base_model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,   # sampling gives more variety
        top_p=0.9,
        temperature=0.8,
        pad_token_id=tok.eos_token_id
    )
    ft_out = ft_model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        pad_token_id=tok.eos_token_id
    )

# === Decode results ===
print("=== Base GPT-2 Large ===")
print(tok.decode(base_out[0], skip_special_tokens=True))

print("\n=== Fine-tuned Investigator ===")
print(tok.decode(ft_out[0], skip_special_tokens=True))
