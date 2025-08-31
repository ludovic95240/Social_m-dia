import networkx as nx
import matplotlib.pyplot as plt

# Charger le graphe
G = nx.read_gexf("App/api/processed/network/interaction_graph.gexf")

# Optionnel : ne garder que les 200 nœuds les plus connectés (plus lisible)
deg = dict(G.degree(weight="weight"))
top_nodes = sorted(deg, key=deg.get, reverse=True)[:200]
H = G.subgraph(top_nodes)

# Layout force-directed
pos = nx.spring_layout(H, seed=42, weight="weight")

plt.figure(figsize=(12, 10))
nx.draw_networkx_nodes(H, pos, node_size=100, alpha=0.7)
nx.draw_networkx_edges(H, pos, alpha=0.4)
nx.draw_networkx_labels(H, pos, font_size=7)
plt.title("Graphe d’interaction Honda (top 200 nœuds)")
plt.axis("off")
plt.savefig("App/api/processed/network/interaction_graph.png")
plt.show()
