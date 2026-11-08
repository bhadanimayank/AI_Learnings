import urllib.request
import os

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
path = "shakespeare.txt"

if not os.path.exists(path):
    urllib.request.urlretrieve(url, path)

with open(path, "r", encoding="utf-8") as f:
    text = f.read()

print("total characters:", len(text))
print(text[:300])   # peek at the first 300 characters

chars = sorted(list(set(text)))     # every unique character, sorted
vocab_size = len(chars)
print("vocabulary:", "".join(chars))
print("vocab size:", vocab_size)

stoi = {ch: i for i, ch in enumerate(chars)}   # string-to-index
itos = {i: ch for i, ch in enumerate(chars)}   # index-to-string

def encode(s):
    return [stoi[c] for c in s]

def decode(indices):
    return "".join(itos[i] for i in indices)

print(encode("hello"))
print(decode(encode("hello")))

import torch

data = torch.tensor(encode(text), dtype=torch.long)
print("data shape:", data.shape)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8

def get_batch(source, batch_size=32):
    ix = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i:i+block_size] for i in ix])
    y = torch.stack([source[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch(train_data)
print("input batch shape:", xb.shape)
print("target batch shape:", yb.shape)
print("example input: ", decode(xb[0].tolist()))
print("example target:", decode(yb[0].tolist()))

import torch.nn as nn
import torch.nn.functional as F

class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.embedding(idx)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

model = BigramModel(vocab_size)
logits, loss = model(xb, yb)
print("logits shape:", logits.shape)
print("loss:", loss.item())

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

loss_history = []
steps = 3000

for step in range(steps):
    xb, yb = get_batch(train_data)
    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())
    if step % 300 == 0:
        print(f"step {step}: loss {loss.item():.3f}")

print("final loss:", loss_history[-1])

def generate(model, start_text, max_new_chars=300):
    idx = torch.tensor([encode(start_text)], dtype=torch.long)   # shape (1, len)

    for _ in range(max_new_chars):
        logits, _ = model(idx[:, -1:])          # bigram only needs the last character
        probs = F.softmax(logits[:, -1, :], dim=-1)   # turn scores into probabilities
        next_char = torch.multinomial(probs, num_samples=1)   # sample randomly per the probabilities
        idx = torch.cat([idx, next_char], dim=1)

    return decode(idx[0].tolist())

print(generate(model, start_text="\n", max_new_chars=300))

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(loss_history)
axes[0].set_title("Training loss")
axes[0].set_xlabel("step")

# visualize the embedding table in 2D using PCA
W = model.embedding.weight.detach().numpy()   # shape (vocab_size, vocab_size)
W_centered = W - W.mean(axis=0)
U, S, Vt = __import__("numpy").linalg.svd(W_centered, full_matrices=False)
W_2d = U[:, :2] * S[:2]                        # project onto top 2 directions

ax = axes[1]
ax.scatter(W_2d[:, 0], W_2d[:, 1])
for i, ch in enumerate(chars):
    label = repr(ch) if ch in ("\n", " ") else ch
    ax.annotate(label, (W_2d[i, 0], W_2d[i, 1]))
ax.set_title("Character embeddings (2D projection)")

plt.tight_layout()
plt.show()