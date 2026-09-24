import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

n = 100
angles = np.random.rand(2 * n) * 2 * np.pi
radii = np.concatenate([np.full(n, 1.0), np.full(n, 2.5)]) + np.random.randn(2 * n) * 0.15
X = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
y = np.array([1] * n + [0] * n)      # inner ring = class 1, outer ring = class 0

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm")
plt.show()


H = 4                              # number of hidden neurons
W1 = np.random.randn(2, H)         # hidden weights: 2 inputs -> 4 neurons
b1 = np.zeros(H)                   # hidden biases
W2 = np.random.randn(H)            # output weights: 4 hidden -> 1 output
b2 = 0.0                           # output bias
lr = 0.5
epochs = 3000
N = len(y)

def sigmoid(s):
    return 1 / (1 + np.exp(-s))

def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1               # shape (200, 4)
    A1 = np.tanh(Z1)               # shape (200, 4)
    p = sigmoid(A1 @ W2 + b2)      # shape (200,)
    return A1, p


loss_history = []
W1_history = []
W2_history = []

for epoch in range(epochs):
    # forward pass
    A1, p = forward(X, W1, b1, W2, b2)
    loss = -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))

    # backward pass (backpropagation)
    error = p - y                          # (200,)
    grad_W2 = A1.T @ error / N             # (4,)
    grad_b2 = error.mean()
    dA1 = np.outer(error, W2)              # (200, 4)
    dZ1 = dA1 * (1 - A1 ** 2)              # (200, 4)
    grad_W1 = X.T @ dZ1 / N                # (2, 4)
    grad_b1 = dZ1.mean(axis=0)             # (4,)

    # record, then update
    loss_history.append(loss)
    W1_history.append(W1.copy())
    W2_history.append(W2.copy())

    W1 = W1 - lr * grad_W1
    b1 = b1 - lr * grad_b1
    W2 = W2 - lr * grad_W2
    b2 = b2 - lr * grad_b2

_, p = forward(X, W1, b1, W2, b2)
print("final loss:", loss_history[-1], "accuracy:", ((p > 0.5) == y).mean())


W1_history = np.array(W1_history)      # (epochs, 2, H)
W2_history = np.array(W2_history)      # (epochs, H)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# --- Panel 1: the decision boundary (colored background) ---
gx, gy = np.meshgrid(np.linspace(-4, 4, 200), np.linspace(-4, 4, 200))
grid = np.column_stack([gx.ravel(), gy.ravel()])
_, p_grid = forward(grid, W1, b1, W2, b2)

ax = axes[0, 0]
ax.contourf(gx, gy, p_grid.reshape(gx.shape), levels=20, cmap="coolwarm", alpha=0.6)
ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=20)
ax.set_title("Network's decision regions")

# --- Panel 2: the line drawn by each hidden neuron ---
ax = axes[0, 1]
ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", alpha=0.5, s=20)
xs = np.array([-4, 4])
for j in range(H):
    ax.plot(xs, -(W1[0, j] * xs + b1[j]) / W1[1, j], label=f"hidden {j+1}")
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.legend()
ax.set_title("Each hidden neuron's line")

# --- Panel 3: loss ---
axes[1, 0].plot(loss_history)
axes[1, 0].set_title("Loss")
axes[1, 0].set_xlabel("epoch")

# --- Panel 4: all weights over time ---
ax = axes[1, 1]
for j in range(H):
    ax.plot(W1_history[:, 0, j], color=f"C{j}", label=f"W1[x1->h{j+1}]")
    ax.plot(W1_history[:, 1, j], color=f"C{j}", linestyle=":", label=f"W1[x2->h{j+1}]")
    ax.plot(W2_history[:, j], color=f"C{j}", linestyle="--", label=f"W2[h{j+1}->out]")
ax.set_title("Weights over time\n(solid: x1 weights, dotted: x2 weights, dashed: output weights)")
ax.set_xlabel("epoch")

plt.tight_layout()
plt.show()