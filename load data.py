import gzip
import csv
import networkx as nx

def load_edges_to_graph(filename):
    G = nx.Graph()
    with gzip.open(filename, mode='rt', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                # Assumes the first two columns are the nodes of the edge
                node1, node2 = row[0], row[1]
                G.add_edge(node1, node2)
    return G

if __name__ == "__main__":
    graph = load_edges_to_graph("edges (1).csv.gz")
    print(f"Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")