import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def chat(model_dir="outputs/checkpoints", max_new_tokens=100):
    """
    中文对话模型推理脚本
    """
    print("🔧 正在加载模型...")

    # 加载tokenizer和模型
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        model.eval()

        # 确保pad_token设置正确
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"📱 设备: {'CUDA' if torch.cuda.is_available() and model.device.type == 'cuda' else 'CPU'}")
        print(f"🔤 词汇表大小: {len(tokenizer)}")

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    print("\n🤖 中文对话模型已加载，输入 'exit' 退出")
    print("💡 提示：输入 'clear' 清除对话历史\n")

    # 维护对话历史（可选）
    conversation_history = []

    while True:
        try:
            user_input = input("\n你：").strip()

            if user_input.lower() in ["exit", "quit", "q", "退出"]:
                print("👋 再见！")
                break

            if user_input.lower() in ["clear", "清除"]:
                conversation_history = []
                print("🗑️ 对话历史已清除")
                continue

            if not user_input:
                continue

            # 构建输入prompt - 关键改进
            # 方式1: 简单的指令格式
            prompt = f"用户：{user_input}\n助手："

            # 方式2: 如果需要对话历史（可选）
            # if conversation_history:
            #     context = "\n".join([f"用户：{h['user']}\n助手：{h['assistant']}" for h in conversation_history[-3:]])  # 保留最近3轮对话
            #     prompt = f"{context}\n用户：{user_input}\n助手："
            # else:
            #     prompt = f"用户：{user_input}\n助手："

            # tokenize输入
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256 // 2  # 为生成留出空间
            )

            if torch.cuda.is_available():
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # 生成回复
            print("🤔 AI正在思考...")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.8,  # 稍微提高创造性
                    top_p=0.85,  # 稍微降低，减少随机性
                    top_k=50,  # 添加top_k采样
                    repetition_penalty=1.1,  # 减少重复
                    no_repeat_ngram_size=3,  # 避免重复3-gram
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    early_stopping=True
                )

            # 解码并处理回复
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 提取AI回复部分
            if "助手：" in full_response:
                reply = full_response.split("助手：")[-1].strip()
            else:
                # 如果没有找到分隔符，移除输入部分
                reply = full_response[len(prompt):].strip()

            # 清理回复
            reply = clean_response(reply)

            if reply:
                print(f"AI：{reply}")

                # 保存对话历史（可选）
                # conversation_history.append({
                #     "user": user_input,
                #     "assistant": reply
                # })
            else:
                print("AI：抱歉，我没能理解您的问题，请换个方式提问。")

        except KeyboardInterrupt:
            print("\n\n👋 用户中断，再见！")
            break
        except Exception as e:
            print(f"❌ 生成过程中出现错误: {e}")
            continue


def clean_response(response):
    """
    清理AI响应
    """
    if not response:
        return ""

    # 移除可能的提示词残留
    unwanted_prefixes = ["用户：", "助手：", "AI：", "你："]
    for prefix in unwanted_prefixes:
        if response.startswith(prefix):
            response = response[len(prefix):].strip()

    # 在句号、问号、感叹号处截断（避免不完整句子）
    for punct in ["。", "！", "？", ".", "!", "?"]:
        if punct in response:
            # 保留到最后一个完整句子
            sentences = response.split(punct)
            if len(sentences) > 1 and sentences[-1].strip() == "":
                response = punct.join(sentences[:-1]) + punct
                break

    # 移除多余的换行和空格
    response = response.replace("\n", " ").strip()

    # 截断过长回复
    if len(response) > 200:
        response = response[:200] + "..."

    return response


def test_specific_prompts():
    """
    测试特定的提示词格式
    """
    model_dir = "outputs/checkpoints"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        model.eval()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        test_cases = [
            "你好",
            "你是谁？",
            "今天天气怎么样？",
            "请介绍一下Python编程语言"
        ]

        for test in test_cases:
            prompt = f"<|user|>{test}<|assistant|>"  # 尝试不同的格式
            inputs = tokenizer(prompt, return_tensors="pt")

            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"测试: {test}")
            print(f"结果: {response}\n")

    except Exception as e:
        print(f"测试失败: {e}")


if __name__ == "__main__":
    # 运行主对话程序
    chat()

    # 如果想要测试不同格式，取消下面的注释
    # print("\n" + "="*50)
    # print("测试不同的prompt格式:")
    # test_specific_prompts()