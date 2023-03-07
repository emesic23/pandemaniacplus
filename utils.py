import numpy as np
import collections
import networkx as nx
import json
from networkx.algorithms import community
from bs4 import BeautifulSoup
import re

def sim_1v1(A, seed1, seed2):
    """
    Simulate a 2-color game on a given graph.
    Keyword arguments:
    A     -- modified (diagonals are 1.5) adjacency matrix form of the graph in column-major order.
    seed1 -- frozenset of indices representing color 1 seed nodes.
    seed2 -- frozenset of indices representing color 2 seed nodes.
    
    Returns:
    count1, count2 -- number of nodes for each color after convergence (or max iterations reached)
    """

    if seed1 == seed2:
        return 0.0, 0.0

    # Construct initial seeding for each color.
    n = A.shape[0]
    curr = np.zeros(n).reshape((-1,1))
    curr[list(seed1)] += 1
    curr[list(seed2)] -= 1

    # Simulate until convergence or max iterations reached
    max_iter = np.random.randint(100, 201)
    iter = 0
    prev = None

    while (prev != curr).any() and iter < max_iter:
        prev = curr
        curr = np.sign(A @ prev)
        iter += 1

    curr = np.array(curr).flatten()
    counts = collections.Counter(curr)
    return counts[1], counts[-1]

def sim_jungle(A, seeds):
    """
    Simulate an n-color game on a given graph.
    Keyword arguments:
    A     -- modified (diagonals are 1.5) adjacency matrix form of the graph in column-major order.
    seeds -- list of frozenets of indices representing seed nodes for each color.
    
    Returns:
    counts -- number of nodes for each color after convergence (or max iterations reached)
    """
    n = A.shape[0]
    num_seeds = len(seeds)
    curr = np.zeros((n, num_seeds))

    # Keep track of total counts to avoid conflicts.
    total  = np.array([list(seed) for seed in seeds]).flatten()
    counts = collections.Counter(total)
    
    # Construct initial seeding for each color.
    for k, seed in enumerate(seeds):
        for node in seed:
            if counts[node] == 1:
                curr[node,:] -= 1
                curr[node,k]  = 1

    # Simulate until convergence or max iterations reached
    max_iter = np.random.randint(100, 201)
    iter = 0
    prev = None

    while (prev != curr).any() and iter < max_iter:
        prev = curr
        curr = np.sign(A @ prev)
        iter += 1

    return [collections.Counter(np.array(curr[:,c]).flatten())[1] for c in range(num_seeds)]


def get_graph_from_file(filename):
    with open(filename, "r") as graphjson:
        adj_list = json.load(graphjson)
    G = nx.Graph()
    for u in adj_list.keys():
        G.add_node(u)
    for u, neighbors in adj_list.items():
        G.add_edges_from([(u, neighbor) for neighbor in neighbors])
    return G
        
def graph_partition(G, num_sections=10):
    node_list = list(G.nodes())
    np.random.shuffle(node_list)
    partition_size = len(node_list) // num_sections
    partition = [node_list[i:i+partition_size] for i in range(0, len(node_list), partition_size)]

# Create a dictionary mapping nodes to their assigned communities
    node_to_community = {}
    for i, community in enumerate(partition):
        for node in community:
            node_to_community[node] = i

    subgraphs = []
    for i in range(num_sections):
        nodes_in_community = [node for node, community_index in node_to_community.items() if community_index == i]
        subgraph = G.subgraph(nodes_in_community)
        subgraphs.append(subgraph)
    
    sections_info = []
    for sg in subgraphs:
        degrees = dict(sg.degree())
        highest_degree_node = max(degrees, key=degrees.get)
        highest_degree_val = degrees[highest_degree_node]
        num_edges = sg.number_of_edges()
        num_nodes = sg.number_of_nodes()
        sections_info.extend([highest_degree_val, num_edges, num_nodes])
    return sections_info, node_to_community
# [highest degree, num_edges, num_nodes, ..., highest deg, num_edges, num_nodes]
def seed_selection(G, node_to_community, num_seeds, sections_known, num_random=0):
    selected = []

    nodes_in_known = [node for node, community_index in node_to_community.items() if community_index in sections_known]
    known_sg = G.subgraph(nodes_in_known)

    degrees = dict(known_sg.degree())
    total_degrees = 2 * known_sg.number_of_edges()
    if total_degrees != 0:
        degrees = {node:degree/total_degrees for node, degree in degrees.items()}
    else:
        degrees = {node:1/known_sg.number_of_nodes() for node, degree in degrees.items()}
    p_degrees = list(degrees.values())

    if len(sections_known) != 0 and len(list(known_sg.nodes())) >= (num_seeds - num_random):
        selected = list(np.random.choice(list(known_sg.nodes()), num_seeds - num_random, replace=False, p=p_degrees))


    while len(selected) != num_seeds:
        random_node = np.random.choice(G.nodes(), 1)[0]
        if random_node not in selected:
            selected.append(random_node)
    return frozenset(selected)

def construct_input(filename, budget):
    with open(filename, 'r') as f:
        html = f.read()

    # Find all <div> tags with class="wrapper collapse"
    soup = BeautifulSoup(html, 'html.parser')
    sections = soup.find_all('div', {'class': 'wrapper collapse', 'id': re.compile('section-*')})

    # Loop through each section and extract the values from the <th> tags with class="fw-normal"
    input = []

    for section in sections:
        max_degree = int(section.find('th', string='max degree').find_next_sibling('th').text)
        num_nodes = int(section.find('th', string='number of nodes').find_next_sibling('th').text)
        total_degrees = int(float(section.find('th', string='total degrees').find_next_sibling('th').text))

        input.extend([max_degree, total_degrees / 2, num_nodes])

    input.append(budget)

    return input
