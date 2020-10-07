import torch
import torch.optim as optim
import torch.nn as nn

torch.manual_seed(1)
X = torch.Tensor([[-1, 1, 2],[1, 1, 1]])
y = torch.Tensor([0, 1, 1])

alpha = 1

class ShallowNet(nn.Module):
    def __init__(self):
        super(ShallowNet, self).__init__()
        self.fc1 = nn.Linear(2,1, bias=False)
    
    def forward(self, X):
        return self.fc1(X)

net = ShallowNet()
print(net)

net.fc1.weight.data = torch.Tensor([[0.1, 0.1]])

print(net(torch.transpose(X,0,1)).squeeze())

optimizer = optim.SGD(net.parameters(), lr=alpha)
optimizer.zero_grad()

criterion = nn.BCEWithLogitsLoss()

for iter in range(100):
    netOutput = net(torch.transpose(X,0,1)).squeeze()
    loss = criterion(netOutput, y)
    
    loss.backward()
    gn = 0
    for f in net.parameters():
        gn = gn + torch.norm(f.grad)
    print("Loss: %f; ||g||: %f" % (loss, gn))
    optimizer.step()
    optimizer.zero_grad()

for f in net.parameters():
    print(f)
