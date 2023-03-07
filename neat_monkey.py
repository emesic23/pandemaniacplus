pfrom utils import *
import numpy as np
import pickle
import networkx as nx
import neat
import argparse
from tqdm import tqdm

# ckpt 46


NUM_ROUNDS = 4
NUM_SEEDS = 10
POINTS_VALUES = [20, 15, 12, 9, 6, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
graph_info = []
G = None
NODE_TO_COMMUNITY = None

# def eval_genomes_ta_g(genomes, config):
#     global graph_info
#     for genome_id, genome in genomes:
#         genome.fitness = 4.0
#         net = neat.nn.FeedForwardNetwork.create(genome, config)
#         for round in range(NUM_ROUNDS):
            
            
# def eval_genomes_ta_o(genomes, config):
#     global graph_info
#     for genome_id, genome in genomes:
#         genome.fitness = 4.0
#         net = neat.nn.FeedForwardNetwork.create(genome, config)
#         for xi, xo in zip(xor_inputs, xor_outputs):
#             output = net.activate(xi)
#             genome.fitness -= (output[0] - xo[0]) ** 2

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
    global NUM_SEEDS
    global current_balance

    diag = 1.5 * np.ones(G.order())
    A = nx.adjacency_matrix(G) + np.diag(diag)
    graph_info, NODE_TO_COMMUNITY = graph_partition(G)
    test = np.array(genomes)[:, 0]
    for genome_id, genome in tqdm(genomes):
        genome.fitness = 0.0
        rand_opps = np.array([genome])
        temp = np.array(genomes)
        temp = temp[:, 1]
        rand_opps = np.concatenate([rand_opps, np.random.choice(temp, 10, replace=False)])
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
                known_info[winners[0]].add(i)
                money[winners[0]] -= bids[winners[1], i]
            if money[0] < 0:
                genome.fitness -= 500
        # print(money)
        seedings = [[seed_selection(G, NODE_TO_COMMUNITY, NUM_SEEDS, known_info[agent]) for agent in range(len(rand_opps))] for i in range(10)]
        for seeding in seedings:
            output = sim_jungle(A, seeding)
            output = np.flip(np.argsort(output))

            for i, idx in enumerate(output):
                if idx == 0:
                    genome.fitness += POINTS_VALUES[i]



def run(config_file):
    global G

    parser = argparse.ArgumentParser(description='A simple example of argparse')
    parser.add_argument('--graph', type=str, help='Graph to use')
    args = parser.parse_args()
    graph_name = 'graphs/snap/pickled/' + args.graph + '.pkl'

    with open(graph_name, 'rb') as file:
        G = pickle.load(file)


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
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    # winner = p.run(eval_genomes_jungle, 300)

    # with open("winning_genome.pkl", "wb") as winning_file:
    #     pickle.dump(winner, winning_file)
    # # Display the winning genome.
    # print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    # winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    # for xi, xo in zip(xor_inputs, xor_outputs):
    #     output = winner_net.activate(xi)
    #     print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))


    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-final-mean')
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))
    winner = p.run(eval_genomes_jungle, 2)
    with open("winning_genome.pkl", "wb") as winning_file:
        pickle.dump(winner, winning_file)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    run("monkey_config.cfg")