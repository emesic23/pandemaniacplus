from utils import *
import numpy as np
import pickle
import networkx as nx
import neat
import argparse
from tqdm import tqdm
from datetime import datetime
# ckpt 46
import multiprocessing
from neat.visualize

NUM_ROUNDS = 4
POINTS_VALUES = [20, 15, 12, 9, 6, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
graph_info = []
G = None
NODE_TO_COMMUNITY = None


def greedy_bid(graph_info, money):
    max_degree = []
    for section in range(0, len(graph_info), 3):
        max_degree.append(graph_info[section])
    max_degree = np.array(max_degree)
    bid = np.zeros(len(graph_info))
    # zero all but top 3 max degrees
    max_degree[np.argsort(max_degree)[:-3]] = 0
    normalized_max_degree = max_degree / np.sum(max_degree)
    bid = normalized_max_degree * money
    return bid

def opportunistic_bid(graph_info, money):
    max_degree = []
    num_nodes = []
    num_edges = []
    for section in range(0, len(graph_info), 3):
        max_degree.append(graph_info[section])
        num_nodes.append(graph_info[section+2])
        num_edges.append(graph_info[section+1])
    max_degree = np.array(max_degree)
    num_nodes = np.array(num_nodes)
    num_edges = np.array(num_edges)
    
    a = 2.357947691000002
    b = 9.147537007594738e-14
    c = 1
    res = a * max_degree + b * num_nodes + c * num_edges
    normalized = res / np.sum(res)
    bid = normalized * money
    return bid

def eval_genomes_ta_general(genomes, config, ta_bid):
    global graph_info
    global G
    global NODE_TO_COMMUNITY
    global current_balance

    NUM_SEEDS = 10
    n_players = 2
    diag = 1.5 * np.ones(G.order())
    A = nx.adjacency_matrix(G) + np.diag(diag)
    graph_info, NODE_TO_COMMUNITY = graph_partition(G)
    for genome_id, genome in tqdm(genomes):
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config) 
        known_info = {i:set() for i in range(n_players)}
        money = np.array([1000 for i in range(n_players)])
        for round in range(NUM_ROUNDS):
            bids = []
        
            # Bid for the network
            curr_info = graph_info.copy()
            curr_info.append(money[0])
            current_balance = money[0]
            bids.append(output_activation(net.activate(curr_info), money[0]))

            # Bid for the greedy TA
            bids.append(ta_bid(graph_info.copy(), money[1]))
            bids = np.array(bids)
            for i, bid in enumerate(bids.T):
                winners = np.flip(np.argsort(bid))
                if bid[winners[0]] >= 1:
                    known_info[winners[0]].add(i)
                    money[winners[0]] -= bids[winners[1], i]
            if money[0] < 0:
                genome.fitness -= 500
        # print(money)
        seedings = [[seed_selection(G, NODE_TO_COMMUNITY, NUM_SEEDS, known_info[agent]) for agent in range(n_players)] for i in range(10)]
        for seeding in seedings:
            output = sim_1v1(A, seeding[0], seeding[1])
            output = np.flip(np.argsort(output))

            for i, idx in enumerate(output):
                if idx == 0:
                    genome.fitness += POINTS_VALUES[i]
            
            
def eval_genomes_ta_g(genomes, config):
    eval_genomes_ta_multi(genomes, config, greedy_bid)

def eval_genomes_ta_o(genomes, config):
    eval_genomes_ta_multi(genomes, config, opportunistic_bid)

def eval_genomes_ta_multi(genomes, config, ta_bid):
    global graph_info
    global G
    global NODE_TO_COMMUNITY
    global current_balance

    NUM_SEEDS = 10
    diag = 1.5 * np.ones(G.order())
    A = nx.adjacency_matrix(G) + np.diag(diag)
    graph_info, NODE_TO_COMMUNITY = graph_partition(G)
    jobs = []
    for genome_id, genome in tqdm(genomes):
        genome.fitness = 0.0
        jobs.append(pool.apply_async(eval_genome_multi, (genome, config, A, graph_info, NODE_TO_COMMUNITY, G, ta_bid, NUM_SEEDS)))

    for job, (ignored_genome_id, genome) in tqdm(zip(jobs, genomes)):
            genome.fitness = job.get()

def eval_genome_multi(genome, config, A, graph_info, NODE_TO_COMMUNITY, G, ta_bid, NUM_SEEDS):
    n_players = 2
    net = neat.nn.FeedForwardNetwork.create(genome, config) 
    known_info = {i:set() for i in range(n_players)}
    money = np.array([1000 for i in range(n_players)])        
    fitness = 0
    
    for round in range(NUM_ROUNDS):
        bids = []
    
        # Bid for the network
        curr_info = graph_info.copy()
        curr_info.append(money[0])
        current_balance = money[0]
        bids.append(output_activation(net.activate(curr_info), money[0]))

        # Bid for the greedy TA
        bids.append(ta_bid(graph_info.copy(), money[1]))

        bids = np.array(bids)
        for i, bid in enumerate(bids.T):
            winners = np.flip(np.argsort(bid))
            if bid[winners[0]] >= 1:
                known_info[winners[0]].add(i)
                money[winners[0]] -= bids[winners[1], i]
        if money[0] < 0:
            fitness -= 500
    # print(money)
    seedings = [[seed_selection(G, NODE_TO_COMMUNITY, NUM_SEEDS, known_info[agent]) for agent in range(n_players)] for i in range(10)]
    for seeding in seedings:
        output = sim_1v1(A, seeding[0], seeding[1])
        output = np.flip(np.argsort(output))

        for i, idx in enumerate(output):
            if idx == 0:
                fitness += POINTS_VALUES[i]

    return fitness


# Figure out how to constraint the bidding.
# Possibly add massive penalty for crossing the current budget constraint. 

def output_activation(x, current_balance):
    # scale the output to ensure that the sum of the bids does not exceed the available balance
    x = [max(bid, 0) for bid in x]
    scale_factor = 0
    if sum(x) != 0:
        scale_factor = current_balance / sum(x)
    return [bid * scale_factor for bid in x]

def eval_genomes_jungle(genomes, config):
    global graph_info
    global G
    global NODE_TO_COMMUNITY
    global current_balance

    NUM_SEEDS = 20

    diag = 1.5 * np.ones(G.order())
    A = nx.adjacency_matrix(G) + np.diag(diag)
    graph_info, NODE_TO_COMMUNITY = graph_partition(G)
    genomes_idx_dict = {i: genomes[i] for i in range(len(genomes))}
    tested = np.zeros(len(genomes))
    curr = 0
    genomes_dict_keys = np.array(list(genomes_idx_dict.keys()))
    for (genome_id, genome) in genomes:
        genome.fitness = 0.0
        
    for genome_idx, (genome_id, genome) in tqdm(genomes_idx_dict.items()):
        if tested[genome_idx]:
            continue
        
        rand_opps = np.array([genome])
        
        rand_opps_idx = np.concatenate([np.array([genome_idx]), np.random.choice(np.setdiff1d(genomes_dict_keys, genome_idx), 10, replace=False)])

        rand_opps = np.array([genomes_idx_dict[i][1] for i in rand_opps_idx])
        nets = [neat.nn.FeedForwardNetwork.create(struct, config) for struct in rand_opps]
        known_info = {agent:set() for agent in range(len(rand_opps))}
        money = np.array([1000 for i in range(len(rand_opps))])
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
                if bid[winners[0]] >= 1:
                    known_info[winners[0]].add(i)
                    money[winners[0]] -= bids[winners[1], i]
        # print(money)
        seedings = [[seed_selection(G, NODE_TO_COMMUNITY, NUM_SEEDS, known_info[agent]) for agent in range(len(rand_opps))] for i in range(10)]
        for seeding in seedings:
            output = sim_jungle(A, seeding)
            output = np.flip(np.argsort(output))
            # [scores * 11]
            for i, idx in enumerate(output):
                cur_genome_idx = rand_opps_idx[idx]
                if not tested[cur_genome_idx]:
                    genomes[cur_genome_idx][1].fitness += POINTS_VALUES[i]

        tested[rand_opps_idx] = 1
            
def run(config_file):
    global G
    print(multiprocessing.cpu_count())
    parser = argparse.ArgumentParser(description='A simple example of argparse')
    parser.add_argument('--graph', type=str, help='Graph to use')
    parser.add_argument('--tower', type=str, help='strategy (j, g, o)')
    parser.add_argument('--gens', type=int, help="number of generations")
    args = parser.parse_args()
    graph_name = 'graphs/snap/pickled/' + args.graph + '.pkl'

    with open(graph_name, 'rb') as file:
        G = pickle.load(file)

    config_file = f"{config_file}.{args.tower}.cfg"
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    # config.genome_config.add_activation('budget_constraint', output_activation)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5, filename_prefix=f"{args.tower}-checkpoint-"))

    # Run for up to 300 generations.
    if args.tower == 'j':  
        winner = p.run(eval_genomes_jungle, args.gens)
    elif args.tower == 'g':
        winner = p.run(eval_genomes_ta_g, args.gens)
    else:
        winner = p.run(eval_genomes_ta_o, args.gens)

    with open(f"winning_genome.{args.tower}.pkl", "wb") as winning_file:
        pickle.dump(winner, winning_file)
    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))



    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-final-mean')
    # p.add_reporter(neat.StdOutReporter(True))
    # stats = neat.StatisticsReporter()
    # p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5))
    # winner = p.run(eval_genomes_jungle_multi, 2)
    # with open("winning_genome.pkl", "wb") as winning_file:
    #     pickle.dump(winner, winning_file)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    # pool = multiprocessing.Pool(processes=1)
    run("monkey_config")