import torch.nn as nn
from torch_geometric.datasets import KarateClub
import torch
import torch_geometric.nn as tgnn
from sklearn.metrics import accuracy_score

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn = tgnn.GCNConv(34, 10)
        self.fc1 = nn.Linear(10, 4)

    def forward(self, x, edge_index):

        hidden = self.gcn(x, edge_index)
        output = torch.relu(self.fc1(hidden))

        return hidden, output
    
# loading dataset
dataset = KarateClub()
graph = dataset[0]
    
model = Model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.09)

epoch_outputs = []
for epoch in range(100):
    optimizer.zero_grad()
    hidden, output = model(graph.x, graph.edge_index)
    loss = criterion(output, graph.y)

    acc = accuracy_score(output.argmax(dim=1), graph.y)
    loss.backward()

    optimizer.step()
    epoch_outputs.append(output.argmax(dim=1))

    if epoch == 0 or (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1} | Loss: {loss:.2f} | Acc: {acc*100:.2f}%')