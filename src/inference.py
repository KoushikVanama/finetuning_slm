from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from config import MODEL_NAME, MAX_LENGTH, OUTPUT_DIR, DATASET

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16, 
    device_map="auto"
)

prompt = "Distance between two points (1,2) and (3,4) is _____. Elaborate on the concept of distance between two points."
messages = [
    {
        "role": "user",
        "content": prompt
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=256
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
