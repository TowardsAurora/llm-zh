import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def chat(model_dir="outputs/checkpoints", max_new_tokens=200):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")

    print("🤖 中文对话模型已加载，输入 'exit' 退出")

    while True:
        user_input = input("你：")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("👋 再见！")
            break

        inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if reply.startswith(user_input):
            reply = reply[len(user_input):]

        print("AI：" + reply.strip())

if __name__ == "__main__":
    chat()
