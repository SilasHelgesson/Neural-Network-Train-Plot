import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

X_MIN, X_MAX = -10, 10
EPOCHS = 2000
POLYNOMIAL_COEFFICIENTS = [4,-2, 0, 4]
LAYERS = 50
NUM_SAMPLES = 50
PAUSE_BETWEEN_EPOCHS = 0.01 # Seconds

class neural_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(1, LAYERS)
        self.out = nn.Linear(LAYERS, 1)
        self.activation = nn.Tanh()
        
    def forward(self, x):
        h = self.activation(self.hidden(x))
        y = self.out(h)
        return y

model = neural_network()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

p = np.polynomial.Polynomial(POLYNOMIAL_COEFFICIENTS)

x = np.linspace(X_MIN, X_MAX, 100)
y = p(x)

indices = np.random.choice(len(x), NUM_SAMPLES, replace=False)
x_sample = x[indices]
y_sample = y[indices]

x_scaled = 2 * (x_sample - X_MIN) / (X_MAX - X_MIN) - 1
Y_MIN, y_max = x.min(), y.max()
y_scaled = 2 * (y_sample - Y_MIN) / (X_MAX - Y_MIN) - 1

x_train = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_scaled, dtype=torch.float32).unsqueeze(1)

plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))

ax1.plot(x, y, label="f(x)")
ax1.scatter(x_sample, y_sample, color='green', marker='x', label="Sample Points")
nn_line, = ax1.plot([], [], '--', color='red', label="NN prediction")
ax1.set_title("Function Approximation")
ax1.set_xlabel("x")
ax1.set_ylabel("f(x)")
ax1.legend()
ax1.grid(True)

loss_line, = ax2.plot([], [], color='blue')
ax2.set_title("Loss Curve")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("MSE Loss")
ax2.grid(True)

loss_history = []

for epoch in range(EPOCHS):
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    loss_history.append(loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    x_scaled_dense = 2 * (x - X_MIN) / (X_MAX - X_MIN) - 1
    x_tensor_dense = torch.tensor(x_scaled_dense, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        y_pred_dense_scaled = model(x_tensor_dense).numpy().flatten()
    y_pred_dense = (y_pred_dense_scaled + 1)/2 * (y_max - Y_MIN) + Y_MIN
    nn_line.set_data(x, y_pred_dense)

    loss_line.set_data(range(1, epoch+2), loss_history)
    ax2.set_xlim(0, EPOCHS)
    ax2.set_ylim(0, max(loss_history) * 1.1)

    plt.pause(PAUSE_BETWEEN_EPOCHS)

plt.ioff()
plt.show()