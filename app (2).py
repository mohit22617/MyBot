# Install required libraries
# !pip install transformers sentencepiece gradio --quiet # This is not needed in the app.py file for deployment

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import gradio as gr

# Load model & tokenizer
model_name = "facebook/blenderbot-1B-distill"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Chat function
def chatbot_response(user_input, history=[]):
    inputs = tokenizer(user_input, return_tensors="pt")
    reply_ids = model.generate(**inputs, max_length=200)
    bot_reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)

    # Append to chat history
    history.append(("You: " + user_input, "Bot: " + bot_reply))
    return history, ""  # returning "" clears the textbox

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("<h2>🤖 Mohit's  ChatBot</h2>")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Type your message here...")
    clear = gr.Button("Clear Chat")

    msg.submit(chatbot_response, [msg, chatbot], [chatbot, msg])
    clear.click(lambda: None, None, chatbot, queue=False)

# Launch the app
if __name__ == "__main__":
    demo.launch()
