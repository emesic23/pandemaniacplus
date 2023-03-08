import networkx as nx
import pickle
import os

def process_graph_file(filename):
    # Open the file for reading
    with open('graphs/snap/raw/' + filename, 'r') as f:
        # Read in the number of nodes
        num_nodes = int(f.readline())
        label_map = dict()

        # Create an empty graph
        G = nx.Graph()
        for node in range(0, num_nodes):
            G.add_node(node)

        # Add edges to the graph
        i = 0

        self = 0

        for line in f:
            try:
                n1, n2 = tuple(map(int, line.strip().split()[:2]))
            except:  
                n1, n2 = tuple(map(int, line.strip().split(',')[:2]))
            
            if n1 not in label_map:
                label_map[n1] = i
                i += 1
            if n2 not in label_map:
                label_map[n2] = i
                i += 1

            if n1 == n2 or G.has_edge(label_map[n1], label_map[n2]):
                self += 1

            G.add_edge(label_map[n1], label_map[n2])

    print(f'Graph: {filename}')
    print(f'Num node: {G.order()}, num edges: {G.number_of_edges() + self}')

    # Pickle the graph
    with open('graphs/snap/pickled/' + filename.split('.')[0] + '.pkl', 'wb+') as f:
        pickle.dump(G, f)

for filename in os.listdir('graphs/snap/raw'):
    process_graph_file(filename)