import json
from torch.utils.data import Dataset

class AlpacaDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=256):
        self.data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item["instruction"]
        input_text = item.get("input", "")
        output = item["output"]

        if input_text:
            prompt = f"用户: {instruction}\n{input_text}\n助手:"
        else:
            prompt = f"用户: {instruction}\n助手:"

        source = self.tokenizer(prompt, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        target = self.tokenizer(output, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")

        source_ids = source["input_ids"].squeeze()
        target_ids = target["input_ids"].squeeze()

        labels = target_ids.clone()
        labels[target["attention_mask"].squeeze() == 0] = -100

        return {
            "input_ids": source_ids,
            "attention_mask": source["attention_mask"].squeeze(),
            "labels": labels,
        }
