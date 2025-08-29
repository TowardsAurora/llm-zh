import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "outputs/checkpoints"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
if torch.cuda.is_available():
    model.to("cuda")
model.eval()

def chat_fn(history, max_new_tokens=200, temperature=0.7, top_p=0.9):
    user_input = history[-1][0]
    inputs = tokenizer(user_input, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p
        )

    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if reply.startswith(user_input):
        reply = reply[len(user_input):]
    history[-1] = (user_input, reply.strip())
    return history

with gr.Blocks() as demo:
    gr.Markdown("## 🤖 中文对话模型 (本地部署版)")
    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(label="输入内容")
    clear = gr.Button("清除对话")

    def user_input(message, history):
        return "", history + [[message, None]]

    msg.submit(user_input, [msg, chatbot], [msg, chatbot], queue=False).then(
        chat_fn, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
