from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from transformers import TrainerCallback
import torch, os, csv

# === Config ===
MODEL_NAME = "/work7/sean/l8b_investigator_sft_checkpoints/checkpoint-846"
REF_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
DATA_FILE = "fineweb_dpo_l8b.jsonl"
OUTPUT_DIR = "/work7/sean/l8b_investigator_dpo_checkpoints"

DEVICE = "cuda" 
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,7"

# === Load dataset ===
dataset = load_dataset("json", data_files=DATA_FILE, split="train")

# Ensure 'prompt' column exists
def add_prompt(example):
    if "prompt" not in example:
        example["prompt"] = f"Suffix:\n{example['suffix']}\nPrefix:\n"
    return example

dataset = dataset.map(add_prompt)

# === Load tokenizer ===
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# === Load models ===
policy_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
ref_model = AutoModelForCausalLM.from_pretrained(REF_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")

# === DPO Config ===
dpo_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=5e-6,
    beta=0.1,
    logging_steps=50,
    save_steps=200,
    eval_steps=None,
    eval_strategy="no",
    warmup_steps=50,
    bf16=True,
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
    train_dataset=dataset,
    eval_dataset=None,
    processing_class=tok,
)

# === CSV Logger ===
class CSVLoggerCallback(TrainerCallback):
    def __init__(self, log_file):
        self.log_file = log_file
        self.header_written = False

    def _write_logs(self, logs):
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        write_header = not os.path.exists(self.log_file) or not self.header_written
        with open(self.log_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=logs.keys())
            if write_header:
                writer.writeheader()
                self.header_written = True
            writer.writerow(logs)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logs["step"] = state.global_step
            logs["type"] = "train"
            self._write_logs(logs)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            metrics["step"] = state.global_step
            metrics["type"] = "eval"
            self._write_logs(metrics)

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