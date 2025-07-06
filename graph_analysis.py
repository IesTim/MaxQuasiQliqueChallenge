import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Increase recursion limit for large graph algorithms if needed
import sys
sys.setrecursionlimit(1000000)

df_edges = pd.read_csv("edges (1).csv.gz", compression='gzip', header=None, names=['source','target'])
print(f"Loaded {len(df_edges)} edges")

# Build an undirected graph (change to DiGraph() if directed)
G = nx.from_pandas_edgelist(df_edges, 'source', 'target', create_using=nx.Graph())
print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
density = nx.density(G)
print(f"Nodes: {num_nodes}\nEdges: {num_edges}\nDensity: {density:.2e}")

# Compute degrees
degrees = np.array([d for n, d in G.degree()])
print(f"Degree: mean={degrees.mean():.2f}, median={np.median(degrees)}, max={degrees.max()}, std={degrees.std():.2f}")

# Degree histogram
degree_counts = Counter(degrees)
xs, ys = zip(*sorted(degree_counts.items()))

# Plot histogram on log-log scale\ nplt.figure(figsize=(6,4))
plt.loglog(xs, ys, marker='o', linestyle='None')
plt.title('Degree Distribution (log-log)')
plt.xlabel('Degree')
plt.ylabel('Count')
plt.grid(True, which='both', ls='--', linewidth=0.5)
plt.tight_layout()
plt.show()

components = list(nx.connected_components(G))
sizes = [len(c) for c in components]
sizes_sorted = sorted(sizes, reverse=True)
print(f"Number of connected components: {len(components)}")
print(f"Largest component size: {sizes_sorted[0]} ({sizes_sorted[0]/num_nodes:.2%} of nodes)")
print("Top 10 component sizes:", sizes_sorted[:10])

# Extract giant component subgraph
giant = G.subgraph(components[sizes.index(sizes_sorted[0])]).copy()

import random
sample_size = 500
nodes = list(giant.nodes())
distances = []
for _ in range(sample_size):
    u, v = random.sample(nodes, 2)
    try:
        d = nx.shortest_path_length(giant, u, v)
        distances.append(d)
    except nx.NetworkXNoPath:
        pass
print(f"Sampled average shortest path length: {np.mean(distances):.2f}")

print(f"Sampled diameter (max distance): {max(distances)}")

avg_clust = nx.average_clustering(G)
transitivity = nx.transitivity(G)
print(f"Average local clustering coefficient: {avg_clust:.4f}")
print(f"Global clustering coefficient (transitivity): {transitivity:.4f}")

# Distribution of local clustering (sample)
clust_samples = list(nx.clustering(G, nodes=random.sample(nodes, sample_size)).values())
plt.figure(figsize=(6,4))
plt.hist(clust_samples, bins=20)
plt.title('Local Clustering Coefficient Distribution (sample)')
plt.xlabel('Clustering coefficient')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

assortativity = nx.degree_assortativity_coefficient(G)
print(f"Degree assortativity coefficient: {assortativity:.4f}")

cores = nx.core_number(G)
core_values = list(cores.values())
print(f"Max core number: {max(core_values)}")

# Core size distribution
dist_cores = Counter(core_values)
cores_x, cores_y = zip(*sorted(dist_cores.items()))
plt.figure(figsize=(6,4))
plt.bar(cores_x, cores_y)
plt.title('k-core Size Distribution')
plt.xlabel('k')
plt.ylabel('Number of nodes')
plt.tight_layout()
plt.show()

# Top 10 by degree centrality
deg_centrality = nx.degree_centrality(G)
top_deg = sorted(deg_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
print("Top 10 nodes by degree centrality:", top_deg)

# PageRank (approximate)
pr = nx.pagerank(G, alpha=0.85, tol=1e-4, max_iter=100)
top_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:10]
print("Top 10 nodes by PageRank:", top_pr)

# Requires `python-louvain` package: pip install python-louvain
from networkx.algorithms.community import louvain_communities

# This returns a list of sets, one per community:
comms = louvain_communities(G, weight=None, resolution=1)
# Turn it into a node→community dict:
partition = {node: cid for cid, com in enumerate(comms) for node in com}

# Then compute sizes:
from collections import Counter
comm_sizes = Counter(partition.values())
print(f"Detected {len(comm_sizes)} communities")
print("Top 10 community sizes:", comm_sizes.most_common(10))

# Compute (small) eigenvalues of Laplacian for insight on connectivity
# Might use scipy.sparse.linalg for large graphs
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh

# Build sparse Laplacian
L = nx.linalg.laplacianmatrix.laplacian_matrix(giant)
# Compute smallest non-zero eigenvalues
eigvals, _ = eigsh(L, k=5, which='SM')
print("Smallest non-zero Laplacian eigenvalues:", eigvals[1:])

# Sample top 100 by degree
top_nodes = [n for n, d in sorted(G.degree(), key=lambda x: x[1], reverse=True)[:100]]
H = G.subgraph(top_nodes)
plt.figure(figsize=(6,6))
pos = nx.spring_layout(H, k=0.5)
nx.draw(H, pos, node_size=20, with_labels=False)
plt.title('Subgraph of Top 100 High-Degree Nodes')
plt.show()