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

    # 加载tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 为中文优化pad_token设置
    if tokenizer.pad_token is None:
        # 对于中文，建议添加专门的pad_token而不是使用eos_token
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 如果添加了新的特殊token，需要调整模型的embedding层
    if tokenizer.pad_token == '[PAD]':
        model.resize_token_embeddings(len(tokenizer))

    # 加载数据集
    train_dataset = AlpacaDataset(train_file, tokenizer, max_length=max_seq_length)

    print(f"📊 训练数据集大小: {len(train_dataset)}")
    print(f"🔤 词汇表大小: {len(tokenizer)}")

    # 优化后的训练参数
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
        report_to="none",

        # 为中文训练添加的优化参数
        gradient_accumulation_steps=4,  # 增加有效batch size
        weight_decay=0.01,  # 防止过拟合
        lr_scheduler_type="cosine",  # 余弦学习率调度
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        save_safetensors=True,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        # 添加数据整理器来处理变长序列
        data_collator=lambda data: {
            'input_ids': torch.stack([f['input_ids'] for f in data]),
            'attention_mask': torch.stack([f['attention_mask'] for f in data]),
            'labels': torch.stack([f['labels'] for f in data])
        }
    )

    # 开始训练
    print("🚀 开始训练...")
    trainer.train()

    # 保存模型
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ 模型已保存到 {output_dir}")

    # 训练后立即测试
    print("\n🧪 快速测试...")
    test_model(model, tokenizer)


def test_model(model, tokenizer):
    """训练后的快速测试"""
    model.eval()
    test_inputs = ["你好", "今天天气", "请介绍一下"]

    for test_input in test_inputs:
        inputs = tokenizer.encode(f"用户：{test_input}\n助手：", return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"输入: {test_input}")
        print(f"输出: {response}\n")


if __name__ == "__main__":
    main()