from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from transformers import TrainerCallback
import torch, os, csv

# === Config ===
MODEL_NAME = "./investigator_checkpoints/checkpoint-507"
REF_MODEL_NAME = "gpt2-large"
DATA_FILE = "dpo_dataset_clean.jsonl"
OUTPUT_DIR = "./investigator_dpo_checkpoints"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Load dataset ===
dataset = load_dataset("json", data_files=DATA_FILE, split="train")

# Ensure 'prompt' column exists
def add_prompt(example):
    if "prompt" not in example:
        example["prompt"] = f"Suffix:\n{example['suffix']}\nPrefix:\n"
    return example

dataset = dataset.map(add_prompt)

# Split train/val
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_ds, val_ds = dataset["train"], dataset["test"]

# === Load tokenizer ===
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# === Load models ===
policy_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
ref_model = AutoModelForCausalLM.from_pretrained(REF_MODEL_NAME).to(DEVICE)

# === DPO Config ===
dpo_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    learning_rate=5e-6,
    beta=0.1,
    logging_steps=50,
    save_steps=200,
    eval_steps=100,
    warmup_steps=50,
    fp16=torch.cuda.is_available(),
    save_total_limit=3,
    remove_unused_columns=False,  # ✅ critical for custom schema
    report_to=[],  # disable wandb unless you want logging
    save_safetensors=True,  # ✅ ensure safetensors format
)

# === DPO Trainer ===
trainer = DPOTrainer(
    model=policy_model,
    ref_model=ref_model,
    args=dpo_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tok,
)

# === CSV Logger ===
class CSVLoggerCallback(TrainerCallback):
    def __init__(self, log_file):
        self.log_file = log_file
        self.header_written = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        write_header = not os.path.exists(self.log_file) or not self.header_written
        with open(self.log_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=logs.keys())
            if write_header:
                writer.writeheader()
                self.header_written = True
            writer.writerow(logs)

metrics_file = os.path.join(OUTPUT_DIR, "training_metrics.csv")
trainer.add_callback(CSVLoggerCallback(metrics_file))

# === Resume logic ===
latest_ckpt = None
if os.path.isdir(OUTPUT_DIR):
    ckpts = [os.path.join(OUTPUT_DIR, d) for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
    if ckpts:
        latest_ckpt = max(ckpts, key=os.path.getmtime)

train_result = trainer.train(resume_from_checkpoint=latest_ckpt if latest_ckpt else None)

# === Print summary of last metrics ===
print("\n[INFO] Training finished.")
print(train_result)