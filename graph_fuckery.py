import networkx as nx
import pickle
import os

def process_graph_file(filename):
    # Open the file for reading
    with open('graphs/snap/raw/' + filename, 'r') as f:
        # Read in the number of nodes
        num_nodes = int(f.readline())

        # Create an empty graph
        G = nx.Graph()

        # Add nodes to the graph
        for node in range(0, num_nodes):
            G.add_node(node)

        # Add edges to the graph
        for line in f:
            try:
                edge = tuple(map(int, line.strip().split()[:2]))
                G.add_edge(*edge)
            except:  
                edge = tuple(map(int, line.strip().split(',')[:2]))
                G.add_edge(*edge)

    # Pickle the graph
    with open('graphs/snap/pickled/' + filename.split('.')[0] + '.pkl', 'wb+') as f:
        pickle.dump(G, f)

for filename in os.listdir('graphs/snap/raw'):
    process_graph_file(filename)