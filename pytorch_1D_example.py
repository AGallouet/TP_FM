# This file is a pytorch example of how to train a single layer neural network
# We are going to train a nueral network to match f(x) = sin(a * x)  

import torch
import numpy as np
import matplotlib.pyplot as plt


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
if not use_cuda:
    print("Warning : CUDA unavailable")


# Multilayer Perceptron, with one hidden layer:
class MLP(torch.nn.Module):
    def __init__(self, hidden = 64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


a = 2 # Bigger a means more oscilations, harder to approximate 
def f(x):
    return torch.sin(a * x)


niter = 10000
batch_size = 128

hidden_dim = 64

# Define the model 
model = MLP(hidden_dim).to(device)  

#Choose optimizer and give it the model parameters
optimizer = torch.optim.Adam(model.parameters())

# Training loop
for k in range(niter):
    # Reset the gradients to zero
    optimizer.zero_grad()

    # Take a batch of samples
    x = 2 * np.pi * torch.rand(batch_size, device = device).unsqueeze(dim = 1)

    # Compute objective function
    y = f(x)

    # Compute loss, mean squared error here
    loss = (model(x) - y).pow(2).sum()

    # loss.backward() computes the gradient of the loss with respect to the parameters
    loss.backward()

    # optimizer.step() updates the parameters
    optimizer.step()

    if k % (niter // 100) == 0:
        print(f"iter {k}, loss = {loss}")


with torch.no_grad():
    x = torch.linspace(0, 2*np.pi, 100, device = device).unsqueeze(dim = 1)
    y = f(x)
    z = model(x)

    plt.plot(x.cpu(),y.cpu(),label = "exact function")
    plt.plot(x.cpu(),z.cpu(),label = "trained model")
    plt.legend()
    plt.show()
