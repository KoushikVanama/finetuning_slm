from transformers import AutoTokenizer, AutoModelForCausalLM

from config import MODEL_NAME, MAX_LENGTH, OUTPUT_DIR, DATASET

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
print(model)

total = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total}")

trainable = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(f"Trainable parameters: {trainable}")

from peft import LoraConfig
from peft import get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "v_proj"
    ]
)

model = get_peft_model(
    model,
    config
)

print(model.print_trainable_parameters())
print(model.named_modules(), "named_modules")