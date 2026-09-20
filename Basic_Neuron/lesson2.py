import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

n = 100
red  = np.random.randn(n, 2) + np.array([-2, -2])
blue = np.random.randn(n, 2) + np.array([ 2,  2])

X = np.vstack([red, blue])
y = np.array([0]*n + [1]*n)

w = np.random.randn(2) * 0.1   # two small random starting weights
b = 0.0                        # bias starts at zero
lr = 0.1                       # learning rate
epochs = 100                   # how many training rounds

def sigmoid(s):
    return 1 / (1 + np.exp(-s))

# we'll record these each epoch so we can plot them later
loss_history = []
w_history = []
b_history = []

for epoch in range(epochs):
    # 1. Forward pass: compute predictions
    scores = X @ w + b
    p = sigmoid(scores)

    # 2. Compute the loss (cross-entropy, averaged over all points)
    loss = -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))

    # 3. Compute the gradients
    error = p - y
    grad_w = X.T @ error / len(y)
    grad_b = error.mean()

    # 4. Record the current state (before changing it)
    loss_history.append(loss)
    w_history.append(w)
    b_history.append(b)

    # 5. Gradient descent: step downhill
    w = w - lr * grad_w
    b = b - lr * grad_b

accuracy = ((sigmoid(X @ w + b) > 0.5) == y).mean()
print("final weights:", w, "bias:", b)
print("final loss:", loss_history[-1], "accuracy:", accuracy)

w_history = np.array(w_history)   # convert list to array, shape (epochs, 2)
b_history = np.array(b_history)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# --- Panel 1: data and the neuron's line ---
ax = axes[0]
ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", alpha=0.6)
xs = np.array([-5, 5])

def draw_line(wt, bias, **style):
    # the line where score = 0:  w1*x1 + w2*x2 + b = 0  ->  x2 = -(w1*x1 + b) / w2
    ax.plot(xs, -(wt[0] * xs + bias) / wt[1], **style)

for e in [0, 3, 10, 30]:                       # early snapshots, faint
    draw_line(w_history[e], b_history[e], color="gray", alpha=0.5, linestyle="--")
draw_line(w, b, color="black", linewidth=2)     # final line, solid

ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_title("Decision line (dashed = early, solid = final)")

# --- Panel 2: loss over time ---
axes[1].plot(loss_history)
axes[1].set_title("Loss")
axes[1].set_xlabel("epoch")

# --- Panel 3: weights over time ---
axes[2].plot(w_history[:, 0], label="w1")
axes[2].plot(w_history[:, 1], label="w2")
axes[2].plot(b_history, label="b")
axes[2].set_title("Weights and bias")
axes[2].set_xlabel("epoch")
axes[2].legend()

plt.tight_layout()
plt.show()