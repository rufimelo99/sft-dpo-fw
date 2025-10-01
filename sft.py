from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
import numpy as np
import torch, json, os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# === Config ===
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
DATA_FILE = "fineweb_train.jsonl"
OUTPUT_DIR = "/work7/sean/l8b_investigator_sft_checkpoints"
MAX_LENGTH = 128     # suffix (64) + prefix (64)
BATCH_SIZE = 16       # adjust for GPU VRAM
EPOCHS = 3
SAVE_STEPS = 250
LOG_STEPS = 50
METRICS_FILE = "train_metrics.jsonl"

# === Load dataset ===
dataset = load_dataset("json", data_files=DATA_FILE, split="train")

# === Tokenizer ===
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# === Preprocess: input = suffix, label = prefix ===
def preprocess(example):
    text = f"<suffix> {example['suffix']} <prefix> {example['prefix']}"
    tokens = tok(text)
    return tokens

train_ds = dataset.map(preprocess, remove_columns=["prefix", "suffix"])

# === Data collator ===
collator = DataCollatorForLanguageModeling(tok, mlm=False)

# === Model ===
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)

# === Training args ===
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=False,
    do_eval=False,
    eval_steps=LOG_STEPS,
    logging_steps=LOG_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    save_safetensors=True,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    warmup_steps=50,
    learning_rate=5e-5,
    bf16=True,
    report_to="none",
)

# === Metrics ===
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(
        logits.view(-1, logits.size(-1)),
        labels.view(-1)
    ).item()

    ppl = np.exp(loss)
    return {"loss": loss, "perplexity": ppl}

# === Callback to save metrics ===
class SaveMetricsCallback(TrainerCallback):
    def __init__(self, path=METRICS_FILE):
        self.path = path
        if not os.path.exists(self.path):
            open(self.path, "w").close()
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            with open(self.path, "a") as f:
                f.write(json.dumps({"step": state.global_step, **metrics}) + "\n")

# === Trainer ===
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

trainer.add_callback(SaveMetricsCallback(METRICS_FILE))

# === Train (resumable) ===
latest_ckpt = None
if os.path.isdir(OUTPUT_DIR):
    # look for subdirectories named "checkpoint-*"
    subdirs = [os.path.join(OUTPUT_DIR, d) for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
    if subdirs:
        latest_ckpt = max(subdirs, key=os.path.getmtime)  # newest checkpoint

trainer.train(resume_from_checkpoint=latest_ckpt)