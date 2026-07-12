from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from config import MODEL_NAME, MAX_LENGTH, OUTPUT_DIR, DATASET

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16, 
    device_map="auto"
)

# prompt = "Distance between two points (1,2) and (3,4) is _____. Elaborate on the concept of distance between two points."
prompt = "Tell me a story about India"
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
print(text, "text")
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(inputs, "inputs")
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.2,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.0,
    do_sample=True,
    # num_beams=4,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
