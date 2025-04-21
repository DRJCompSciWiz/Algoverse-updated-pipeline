from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)

def generate_code(prompt, max_length=300, temperature=0.2):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

prompt = "physics prompt TBD"
generated_code = generate_code(prompt)
print(generated_code)

#You'll need to import torch, huggingface, & transformers
#Enter in terminal 'pip install transformers torch huggingface_hub'
