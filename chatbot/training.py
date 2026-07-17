import torch
import torch.nn as nn
from torch.nn import functional as F

# ==========================================================
# 1. HYPERPARAMETERS & SYSTEM SETUP
# ==========================================================
batch_size = 32        # How many independent sequences will we process in parallel?
block_size = 64        # What is the maximum context length for predictions?
max_iters = 2000       # Total training epochs/iterations
eval_interval = 200    # How often to check validation loss
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 64            # Embedding dimensions
n_head = 4             # Multi-head attention heads
n_layer = 4            # Total Transformer layers

# Performance optimization for CPU execution
if device == 'cpu':
    torch.set_num_threads(4) # Maximizes compute utilization across cores

print(f"Using execution device: {device}")

# ==========================================================
# 2. PREPARE YOUR GROCERY TRAINING CORPUS
# ==========================================================
print("Loading popular Instacart dataset for training...")
try:
    with open('popular_groceries_training.txt', 'r', encoding='utf-8') as f:
        data_text = f.read()
except FileNotFoundError:
    print("Error: Run prepare_data.py first to generate the text file!")
    exit()

# Extract unique characters (our vocabulary)
chars = sorted(list(set(data_text)))
vocab_size = len(chars)

# Map characters to integers (Tokenization)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] 
decode = lambda l: ''.join([itos[i] for i in l])

# Convert dataset text into numerical tensors
data = torch.tensor(encode(data_text), dtype=torch.long)
n = int(0.9 * len(data)) # 90% train, 10% validation split
train_data = data[:n]
val_data = data[n:]

# BATCH DATA LOADER UTILITY
def get_batch(split):
    dataset = train_data if split == 'train' else val_data
    # Generate random batch starting indices within safety boundaries
    ix = torch.randint(len(dataset) - block_size, (batch_size,))
    # Pack sequential block window contexts and target shift labels
    x = torch.stack([dataset[i:i+block_size] for i in ix])
    y = torch.stack([dataset[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# ==========================================================
# 3. BUILD THE TRANSFORMER ARCHITECTURE
# ==========================================================
class Head(nn.Module):
    """ One head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        
        # Compute attention scores (affinities)
        wei = q @ k.transpose(-2, -1) * C**-0.5 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # Causal mask
        wei = F.softmax(wei, dim=-1)
        
        # Perform weighted aggregation of values
        v = self.value(x) 
        out = wei @ v 
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention running in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # Residual connections
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

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) 
        x = tok_emb + pos_emb 
        x = self.blocks(x) 
        x = self.ln_f(x)
        logits = self.lm_head(x) 

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # Crop context window to size bounds
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # Focus on last predictive step
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ==========================================================
# 4. MODEL INITIALIZATION & TRAINING LOOP
# ==========================================================
model = MiniLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("Starting custom LLM training framework...")
for iter in range(max_iters):
    # Periodically track loss metrics on the train and val sets
    if iter % eval_interval == 0:
        model.eval()
        with torch.no_grad():
            xb, yb = get_batch('val') 
            _, loss = model(xb, yb)
            print(f"Step {iter}: Validation Loss = {loss.item():.4f}")
        model.train()

    # Regular training step batch forward/backward passes
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# ==========================================================
# 5. TEXT GENERATION (INFERENCE)
# ==========================================================
model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device) # Start text generation tracking with a newline token
print("\n--- Model Generation Output ---")
print(decode(model.generate(context, max_new_tokens=150)[0].tolist()))

# Add this at the very end of train_llm.py
print("Saving model weights to 'chatbot-custom.pt'...")
torch.save(model.state_dict(), 'chatbot-custom.pt')

# Also save the character mapping metadata so the inference script knows the vocabulary
import json
meta = {'stoi': stoi, 'itos': itos, 'vocab_size': vocab_size}
with open('model_meta.json', 'w') as f:
    json.dump(meta, f)
print("Model check-pointing complete!")