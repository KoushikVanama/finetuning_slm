from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
)
from trl import SFTTrainer

# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_PATH = "data/train.jsonl"
OUTPUT_DIR = "outputs"

# -----------------------------
# Load Dataset
# -----------------------------
dataset = load_dataset(
    "json",
    data_files=DATASET_PATH
)

# -----------------------------
# Load Tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Some models don't define a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -----------------------------
# Load Base Model
# -----------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

# -----------------------------
# LoRA Configuration
# -----------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "v_proj",
    ],
)

# Inject LoRA
model = get_peft_model(model, lora_config)

# Verify trainable parameters
model.print_trainable_parameters()

# -----------------------------
# Training Arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=50,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    weight_decay=0.01,
    logging_steps=1,
    save_strategy="epoch",
    report_to="none",
)

# -----------------------------
# Trainer
# -----------------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    args=training_args,
    processing_class=tokenizer,
)

# -----------------------------
# Train
# -----------------------------
trainer.train()

# -----------------------------
# Save Adapter
# -----------------------------
trainer.save_model(OUTPUT_DIR)

print("\nTraining completed successfully!")
