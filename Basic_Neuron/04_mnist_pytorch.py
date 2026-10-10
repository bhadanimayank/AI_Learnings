import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

transform = transforms.ToTensor()   # converts images to PyTorch tensors, pixel values 0..1

train_data = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_data  = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

print("training images:", len(train_data))
print("test images:", len(test_data))

image, label = train_data[0]
print("image shape:", image.shape)   # should be [1, 28, 28]: 1 color channel, 28x28 pixels
print("label:", label)

plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"label: {label}")
plt.show()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(784, 64)
        self.tanh = nn.Tanh()
        self.output = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.hidden(x)
        x = self.tanh(x)
        x = self.output(x)
        return x

model = Net()
print(model)

from torch.utils.data import DataLoader

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=64, shuffle=False)

loss_fn = nn.CrossEntropyLoss()                        # softmax + cross-entropy together
optimizer = torch.optim.SGD(model.parameters(), lr=0.5) # plain gradient descent, same as before

train_acc_history = []
test_acc_history = []

def accuracy(loader):
    correct, total = 0, 0
    with torch.no_grad():                 # we're only checking, not training, so skip gradient tracking
        for images, labels in loader:
            outputs = model(images)
            predicted = outputs.argmax(dim=1)   # index of the highest score = the guess
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total

epochs = 10
for epoch in range(epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()              # clear old gradients
        outputs = model(images)            # forward pass
        loss = loss_fn(outputs, labels)    # compute loss
        loss.backward()                    # backward pass: autograd computes all gradients
        optimizer.step()                   # update all parameters

    train_acc = accuracy(train_loader)
    test_acc = accuracy(test_loader)
    train_acc_history.append(train_acc)
    test_acc_history.append(test_acc)
    print(f"epoch {epoch+1}: train acc {train_acc:.3f}, test acc {test_acc:.3f}")

plt.figure(figsize=(6, 4))
plt.plot(train_acc_history, label="train accuracy")
plt.plot(test_acc_history, label="test accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.title("Training vs test accuracy")
plt.show()

# look at a few predictions
images, labels = next(iter(test_loader))
outputs = model(images)
predicted = outputs.argmax(dim=1)

fig, axes = plt.subplots(1, 8, figsize=(14, 2))
for i in range(8):
    axes[i].imshow(images[i].squeeze(), cmap="gray")
    color = "green" if predicted[i] == labels[i] else "red"
    axes[i].set_title(f"{predicted[i].item()}", color=color)
    axes[i].axis("off")
plt.show()