from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_ID = "sshleifer/tiny-gpt2"

def main():
    device = (
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        use_safetensors=True,          # <- critical: avoids torch.load on .bin
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,
    ).to(device)

    prompts = [
        "Once upon a time",
        "In a shocking discovery",
        "The quick brown fox",
        "As a software engineer, I",
        "In Japan, the Shinkansen",
    ]

    for i, p in enumerate(prompts, 1):
        torch.manual_seed(100 + i)
        x = tok(p, return_tensors="pt").to(device)
        y = model.generate(
            **x, max_new_tokens=40, do_sample=True, top_p=0.95, temperature=0.9,
            pad_token_id=tok.eos_token_id
        )
        print(f"\n=== Output {i} ===\n{tok.decode(y[0], skip_special_tokens=True)}")

if __name__ == "__main__":
    main()
