import torch.nn as nn
import torch

import matplotlib.pyplot as plt

from model import LinearModel, LinearModelV2

def plot_two(x1, y1, x2, y2, name):
    plt.figure(figsize=(10,7))
    plt.scatter(x1, y1, c='b', s=4)
    plt.scatter(x2, y2, c='g', s=4)
    plt.savefig(name)

# Select device, cuda if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ax + b, where a is weight and b i bias
weight = 0.3
bias = 0.9

# Create data
start = -2
stop = 0
step = 0.05 # 2 / 0.05 = 100 datapoint
x = torch.arange(start, stop, step).unsqueeze(dim=1)
y = weight * x + bias

# Split data into train and test 80/20
train_size = int(0.8 * len(x))
x_train, y_train = x[:train_size], y[:train_size]
x_test, y_test = x[train_size:], y[train_size:]

# Plot data
plot_two(x_train, y_train, x_test, y_test, "input-data.png")

# Create model
torch.manual_seed(777)
model = LinearModelV2()
model.to(device)
print(next(model.parameters()).device)

# Train model
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001)

epochs = 10000

# Move data to correct device
x_train = x_train.to(device)
y_train = y_train.to(device)
x_test = x_test.to(device)
y_test = y_test.to(device)

for epoch in range(epochs):
    model.train()

    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()

    with torch.inference_mode():
        test_pred = model(x_test)
        test_loss = loss_fn(test_pred, y_test)
# Comment this out for measuring execution time
#        if epoch % 20 == 0:
#            print(f"Epoch: {epoch} | MAE train loss: {loss} | MAE test loss: {test_loss}")

# Model parameters
print("Model parameters:")
print(model.state_dict())
print("The orginal parameters:")
print(f"weight: {weight}, bias {bias}")
