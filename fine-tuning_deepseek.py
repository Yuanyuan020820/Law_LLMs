from datasets import load_dataset, Dataset
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_jsonl_to_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    # extract prompt-response pairs
    examples = []
    for item in data:
        messages = item["messages"]
        user_prompt = ""
        assistant_response = ""
        for msg in messages:
            if msg["role"] == "user":
                user_prompt = msg["content"]
            elif msg["role"] == "assistant":
                assistant_response = msg["content"]
        examples.append({"prompt": user_prompt, "response": assistant_response})
    return Dataset.from_list(examples)

dataset = load_jsonl_to_dataset("uk_immigration_finetune.jsonl")


model_name = "deepseek-ai/deepseek-llm-7b-base"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto", trust_remote_code=True)
