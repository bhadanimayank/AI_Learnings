import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)          # makes the "random" numbers the same every run

n = 100                    # points per group
red  = np.random.randn(n, 2) + np.array([-2, -2])   # cluster around (-2, -2)
blue = np.random.randn(n, 2) + np.array([ 2,  2])   # cluster around ( 2,  2)

X = np.vstack([red, blue])          # stack them into one array, shape (200, 2)
y = np.array([0]*n + [1]*n)         # labels: 0 = red, 1 = blue

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm")
plt.show()

w = np.array([1.0, 1.0])   # weights
b = 0                    # bias

scores = X @ w + b                       # the score for all 200 points at once
predictions = (scores > 0).astype(int)   # 1 if score > 0, else 0

accuracy = (predictions == y).mean()
print("accuracy:", accuracy)