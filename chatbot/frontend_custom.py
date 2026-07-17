# frontend.py
import gradio as gr
import requests

BACKEND_URL = "http://127.0.0.1:8000/chat"

def predict(message, history):
    payload = {"message": message}
    try:
        response = requests.post(BACKEND_URL, json=payload, timeout=10)
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"Backend Server Error Status: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "Network Failure: Unable to handshake with the backend port engine. Did you launch backend.py?"

# Layout Render Pipeline Engine Loop
with gr.Blocks(theme="soft") as demo:
    gr.ChatInterface(
        fn=predict, 
        title="🧠 Custom Transformer Local Assistant",
        description=(
            "Welcome! This chat interface communicates directly with your custom-built language model "
            "running locally on PyTorch. Type the name of a popular grocery item (e.g., 'Banana', "
            "'Organic Strawberries', 'Sparkling Water') to test its structural text prediction capabilities."
        )
    )

if __name__ == "__main__":
    # Boot frontend server layer loop
    demo.launch(server_name="127.0.0.1", server_port=7860)