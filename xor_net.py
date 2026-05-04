from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

X = torch.tensor([
    [0., 0.],
    [1., 0.],
    [0., 1.],
    [1., 1.]
])

Y = torch.tensor([
    [0.],
    [1.],
    [1.],
    [0.]
])

class XOR_net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(2, 4)
        self.layer2 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        return x
    
model = XOR_net()
criterion = nn.MSELoss()
# criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range (10000):
    # forward pass
    y_pred = model(X)

    #loss
    loss = criterion(y_pred, Y)

    #backward
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    # if(epoch == 1000):
    #     for name, params in model.named_parameters():
    #         print(name, params.grad)

    if(epoch % 1000 == 0): 
        print(f"Epoch: {epoch} | loss: {loss.item()}")

# for name, params in model.named_parameters():
#     print(name, params.grad)

y_final = model(X)
print(y_final)