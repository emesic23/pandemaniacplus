import numpy as np
import collections
import networkx as nx
import json
from networkx.algorithms import community
from bs4 import BeautifulSoup
import re
import math
from datetime import datetime

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
    max_iter = 10 #np.random.randint(100, 201)
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
        degrees = {node : G.degree(node) for node in sg.nodes()}
        highest_degree_node = max(degrees, key=degrees.get)
        highest_degree_val = degrees[highest_degree_node]
        num_nodes = sg.number_of_nodes()
        #num_edges = sg.number_of_edges()
        num_edges = sg.number_of_edges() + np.sum([G.degree[x] - sg.degree[x] for x in sg.nodes()])
        sections_info.extend([highest_degree_val, num_nodes, num_edges])
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
        total_degrees = int(float(section.find('th', string='unique edges (.5 = self edge)').find_next_sibling('th').text))

        input.extend([max_degree, total_degrees / 2, num_nodes])

    input.append(budget)

    return input

def construct_params(filename):
    with open(filename, 'r') as f:
        html = f.read()

    # Find all <div> tags with class="wrapper collapse"
    soup = BeautifulSoup(html, 'html.parser')
    sections = soup.find_all('div', {'class': 'wrapper collapse', 'id': re.compile('section-*')})

    # Loop through each section and extract the values from the <th> tags with class="fw-normal"
    max_degs = []
    num_nodes = []
    num_edges = []

    for section in sections:
        max_degs.append(int(section.find('th', string='max degree').find_next_sibling('th').text))
        num_nodes.append(int(section.find('th', string='number of nodes').find_next_sibling('th').text))
        num_edges.append(int(float(section.find('th', string='unique edges (.5 = self edge)').find_next_sibling('th').text)))

    return np.array(max_degs), np.array(num_nodes), np.array(num_edges)

def assemble_info(graph, results):
    filename_graph = 'graph_info_s1/' + graph + '.html'
    filename_results = 'res_info_s1/' + results + '.html'

    # Read in graph and results info from html files.
    with open(filename_graph, 'r') as f:
        html_graph = f.read()
    with open(filename_results, 'r') as f:
        html_results = f.read()

    input = []

    # Process graph info to determine basic info for each graph.
    soup_graph = BeautifulSoup(html_graph, 'html.parser')
    sections = soup_graph.find_all('div', {'class': 'wrapper collapse', 'id': re.compile('section-*')})

    for section in sections:
        max_deg = int(section.find('th', string='max degree').find_next_sibling('th').text)
        num_nodes = int(section.find('th', string='number of nodes').find_next_sibling('th').text)
        num_edges = int(float(section.find('th', string='unique edges (.5 = self edge)').find_next_sibling('th').text))

        input.extend([max_deg, num_nodes, num_edges])

    # Now look through each round in the results page to figure out rest of important info.
    graph_type = graph[0].upper()
    if graph_type == 'O' or graph_type == 'G':
        size = 2
    elif graph_type == 'J':
        size = 24
    else:
        raise ValueError('Graph type is wrong')
    
    moneys = 1000 * np.ones(size)
    placements = np.zeros(size)
    sections = [set() for _ in range(size)]
    bids = np.zeros((size, 10))

    soup_results = BeautifulSoup(html_results, 'html.parser')
    rounds = soup_results.find_all('div', {'class' : 'wrapper collapse', 'id' : re.compile('round*')})
    round_num = 1

    for round in rounds:
        # First figure out if we've gone too far in the page (round hasn't occurred yet).
        h5_tag = round.find('h5', string=graph)
        if h5_tag is None:
            break

        # If so, parse first table for each team's bid and final clearing prices in this round.
        table_tag = h5_tag.find_next_sibling('table')
        tbody_tag = table_tag.find('tbody')
        tr_tags = tbody_tag.find_all('tr')

        i = 1
        for row in tr_tags:
            td_tags = row.find_all('td')
            name = td_tags[0].text.strip()

            if name == 'ModeloTime':
                bids[0,:] = np.array([int(td.text.strip()) for td in td_tags[1:]])
            elif name == 'Clearing Price':
                clearing_prices = [int(td.text.strip()) for td in td_tags[1:]]
            else:
                bids[i,:] = np.array([int(td.text.strip()) for td in td_tags[1:]])
                i += 1

        #print(f'Bids: {bids}')
        #print(f'Round {round_num} clearing prices: {clearing_prices}')

        # Move on to results for each team.
        table_tag = table_tag.find_next_sibling('table')
        tbody_tag = table_tag.find('tbody')

        # Look at ours first so we don't confuse the inputs.
        our_tag = tbody_tag.find('td', string='ModeloTime')
        our_placement_td = our_tag.find_next_sibling('td')
        our_sections = set([int(x) for x in our_placement_td.find_next_sibling('td').text.split(',')])

        #print(f'Round {round_num} our sections: {our_sections}')

        placements[0] = int(our_placement_td.text.strip())
        #print(f'Sections: {our_sections}')
        sections[0] = our_sections
        for section in our_sections:
            if bids[0,section] > 0 and bids[0,section] == np.amax(bids[:,section]):
                moneys[0] -= clearing_prices[section]

        # Now find money and placement for each other team.
        i = 1

        teams = tbody_tag.find_all('tr')
        for team in teams:
            info = team.find_all('td')
            name = info[0].text.strip()

            if name != 'ModeloTime':
                placements[i] = int(info[1].text.strip())
                their_sections = info[2].text.split(',')
                if their_sections[0] != '':
                    their_sections = set([int(x) for x in their_sections])
                    #print(f'their sections: {their_sections}')
                    sections[i] = their_sections
                    for section in their_sections:
                        if bids[i,section] > 0 and bids[i,section] == np.amax(bids[:,section]):
                            moneys[i] -= clearing_prices[section]

                i += 1

        round_num += 1 
    #   print(f'Moneys: {moneys}')
    #print(f'Placements: {placements}')

    # Add money for each team as well as whether we know each section.
    input.extend(moneys)
    input.extend([x in sections[0] for x in range(10)])

    # Calculate how many other teams know each section.
    section_counts = collections.Counter([item for sublist in sections[1:] for item in sublist])
    input.extend([section_counts[i] for i in range(10)])
    input.append(round_num)

    #print(f'size: {len(input)}')

    return input


def eval_genomes_jungle_multi(genomes, config):
    global graph_info
    global G
    global NODE_TO_COMMUNITY
    global NUM_SEEDS
    global current_balance

    diag = 1.5 * np.ones(G.order())
    A = nx.adjacency_matrix(G) + np.diag(diag)
    graph_info, NODE_TO_COMMUNITY = graph_partition(G)
    jobs = []
    for genome_id, genome in tqdm(genomes):
        genome.fitness = 0.0
        rand_opps = np.array([genome])
        temp = np.array(genomes)
        temp = temp[:, 1]
        rand_opps = np.concatenate([rand_opps, np.random.choice(temp, 10, replace=False)])
        jobs.append(pool.apply_async(eval_genome_multi, (rand_opps, config, A, graph_info, NODE_TO_COMMUNITY, G)))

    for job, (ignored_genome_id, genome) in tqdm(zip(jobs, genomes)):
            genome.fitness = job.get()

def eval_genome_multi(rand_opps, config, A, graph_info, NODE_TO_COMMUNITY, G):
    nets = [neat.nn.FeedForwardNetwork.create(struct, config) for struct in rand_opps]
    known_info = {agent:set() for agent in range(len(rand_opps))}
    money = np.array([1000 for i in range(len(rand_opps))])
    fitness = 0
    for round in range(NUM_ROUNDS):
        bids = []
        for i, net in enumerate(nets):
            curr_info = graph_info.copy()
            curr_info.append(money[i])
            current_balance = money[i]
            # print(net.activate(curr_info))
            bids.append(output_activation(net.activate(curr_info), money[i]))
        bids = np.array(bids)
        for i, bid in enumerate(bids.T):
            winners = np.argsort(bid)
            known_info[winners[0]].add(i)
            money[winners[0]] -= bids[winners[1], i]
        if money[0] < 0:
            fitness -= 500
    # print(money)
    seedings = [[seed_selection(G, NODE_TO_COMMUNITY, NUM_SEEDS, known_info[agent]) for agent in range(len(rand_opps))] for i in range(10)]
    for seeding in seedings:
        output = sim_jungle(A, seeding)
        output = np.flip(np.argsort(output))

        for i, idx in enumerate(output):
            if idx == 0:
                fitness += POINTS_VALUES[i]
    return fitness

def sigmoid(x):
  try:
      return 1 / (1 + math.exp(-x))
  except OverflowError:
      return 0.0 if x < 0 else 1.0