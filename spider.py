import pandas as pd
import json

df = pd.read_csv("uk_immigration_qa_dataset.csv")
output_file = "uk_immigration_finetune.jsonl"

with open(output_file, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        item = {
            "messages": [
                {"role": "system", "content": "You are an expert in UK immigration law."},
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]}
            ]
        }
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
