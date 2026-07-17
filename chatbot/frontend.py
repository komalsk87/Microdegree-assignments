# frontend.py
import gradio as gr
import requests

BACKEND_URL = "http://127.0.0.1:8000/chat"

def predict(message, history):
    payload = {"message": message}
    try:
        response = requests.post(BACKEND_URL, json=payload)
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"Error: Backend returned status code {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to the Backend API. Is backend.py running?"

with gr.Blocks(theme="soft") as demo:
    gr.ChatInterface(
        fn=predict, 
        title="🛒 Instacart Grocery AI Assistant",
        description="Ask me about product store availability or aisle locations! Try asking: 'Where can I find strawberries?' or 'Is organic whole milk in stock?'"
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)