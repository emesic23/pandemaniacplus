from utils import *
import numpy as np
import networkx as nx
import neat

NUM_ROUNDS = 4
NUM_SEEDS = 10
POINTS_VALUES = [20, 15, 12, 9, 6, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
graph_info = []
def eval_genomes_ta_g(genomes, config):
    global graph_info
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for round in range(NUM_ROUNDS):
            
            
def eval_genomes_ta_o(genomes, config):
    global graph_info
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = net.activate(xi)
            genome.fitness -= (output[0] - xo[0]) ** 2

# For each genome, pick 10 other random genomes.
# Have them bid through 4 rounds
# Have them select nodes 50 times
# Run the sim for every round and set the fitness to the total jungle points. 
def eval_genomes_jungle(genomes, config):
    global graph_info
    global G
    global node_to_community
    global num_seeds
    
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        rand_opps = np.array(genome)
        rand_opps = np.concatenate(rand_opps, np.random.choice(genomes, 10, replace=False))
        nets = [neat.nn.FeedForwardNetwork.create(struct, config) for struct in rand_opps]
        known_info = {agent:set() for agent in range(len(rand_opps))}
        money = np.array([1000 for i in range(len(rand_opps))])
        for round in range(NUM_ROUNDS):
            bids = []
            for net in nets:
                bids.append(net.activate(graph_info))
            bids = np.array(bids)
            winners = np.argmax(bids, axis=0)
            for i, winner in enumerate(winners):
                known_info[winner].add(i)
                money[winner] -= bids[winner, i]
        seedings = [frozenset(seed_selection(G, node_to_community, num_seeds, known_info[agent]) for agent in range(len(rand_opps)))]
        output = sim_jungle(A, seedings)
        output = np.flip(np.argsort(output))

        for i, idx in enumerate(output):
            if idx == 0:
                genome.fitness += POINTS_VALUES[i]
        


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))


    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)