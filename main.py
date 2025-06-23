from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt

dataset = KarateClub()

print("Number of graphs: ", len(dataset))
print("Number of features: ", dataset.num_features)
print("Number of classes: ", dataset.num_classes)

graph = dataset[0]

print("Is directed? ", graph.is_directed)
print("Self loops: ", graph.has_self_loops)
print("Isolated nodes: ", graph.has_isolated_nodes)

print(graph.x.shape)
print(graph.x)

G = to_networkx(graph, to_undirected=True)

plt.figure(figsize=(9,6))
nx.draw_networkx(G, 
                 pos=nx.spring_layout(G, seed=42),
                 node_size=400,
                 node_color=graph.y,
                 cmap="Set2",
                 edge_color="orange",
                 font_size=10,
                 font_color = "black")

plt.show()