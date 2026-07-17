# backend.py
import torch
import torch.nn as nn
from torch.nn import functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json

app = FastAPI(title="Custom GPT Engine Backend")

# ==========================================================
# 1. LOAD CONFIGURATION & CONFIG MATRIX
# ==========================================================
try:
    with open('model_meta.json', 'r', encoding='utf-8') as f:
        meta = json.load(f)
except FileNotFoundError:
    print("Error: Missing 'model_meta.json'. Ensure you saved the metadata mapping dictionary.")
    exit()

stoi = meta['stoi']
itos = {int(k): v for k, v in meta['itos'].items()}
vocab_size = meta['vocab_size']

# Hyperparameters (Must match train_llm.py perfectly)
block_size = 64
n_embd = 64
n_head = 4
n_layer = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cpu':
    torch.set_num_threads(4)

# ==========================================================
# 2. TRANSFORMER CORE SKELETON
# ==========================================================
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k, q = self.key(x), self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        return wei @ self.value(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        
    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MiniLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(torch.arange(T, device=device))
        return self.lm_head(self.ln_f(self.blocks(x)))

# ==========================================================
# 3. INITIALIZE STATE WEIGHTS
# ==========================================================
print("Loading custom weights ('chatbot-custom.pt')...")
model = MiniLanguageModel().to(device)
try:
    model.load_state_dict(torch.load('chatbot-custom.pt', map_location=device))
    model.eval()
    print("Neural network layers operational!")
except FileNotFoundError:
    print("Error: Missing weights checkpoint file 'chatbot-custom.pt' in directory.")
    exit()

# ==========================================================
# 4. API ROUTING PIPELINE
# ==========================================================
class ChatRequest(BaseModel):
    message: str

def generate_response(prompt: str, max_tokens=80) -> str:
    encoded = [stoi[c] for c in prompt if c in stoi]
    if not encoded:
        encoded = [stoi.get('\n', 0)]
        
    idx = torch.tensor([encoded], dtype=torch.long, device=device)
    
    for _ in range(max_tokens):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        
        # Stop generating early if model finishes the record line entry
        if idx_next.item() == stoi.get('\n', -1):
            break
            
    response_text = ''.join([itos[i] for i in idx[0].tolist()])
    return response_text[len(prompt):].strip()

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Reconstruct context template format matching training data blocks
        formatted_context = f"Product: {request.message} |"
        with torch.no_grad():
            output_reply = generate_response(formatted_context)
            
        if not output_reply:
            output_reply = "Aisle: Context sequence generation returned empty space structural state."
            
        return {"response": output_reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 