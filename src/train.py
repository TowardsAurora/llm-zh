import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from dataset import AlpacaDataset

def main():
    with open("configs/train_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    model_name = config["model_name"]
    train_file = config["train_file"]
    output_dir = config["output_dir"]
    max_seq_length = config["max_seq_length"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    train_dataset = AlpacaDataset(train_file, tokenizer, max_length=max_seq_length)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        num_train_epochs=config["num_train_epochs"],
        learning_rate=config["learning_rate"],
        warmup_steps=config["warmup_steps"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        fp16=torch.cuda.is_available(),
        logging_dir=os.path.join(output_dir, "logs"),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ 模型已保存到 {output_dir}")

if __name__ == "__main__":
    main()
