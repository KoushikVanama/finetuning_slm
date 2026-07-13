from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

from config import MODEL_NAME, OUTPUT_DIR

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

model = PeftModel.from_pretrained(base_model, "outputs")

prompt = "What does Sai Technologies do?"
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
