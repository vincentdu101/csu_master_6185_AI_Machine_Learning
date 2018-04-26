# -*- coding: utf-8 -*-
import random
import numpy as np

from deap import base
from deap import creator
from deap import tools

# create the fitness base and weight
# individual specifies the groups of data input representing each 
# group of chromosomes in array of array
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Chromosome", list, fitness=creator.FitnessMax)

# specifies the toolbox and the population data generation settings
# populates groups with random int data between 0 and 9
toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 0, 9)
toolbox.register("chromosome", tools.initRepeat, creator.Chromosome, toolbox.attr_int, 8)
toolbox.register("population", tools.initRepeat, list, toolbox.chromosome)

# fitness function that determines the output value of the input group of 
# chromosomes
def evaluateFitnessFunction(ind):
    value = (ind[0] + ind[1]) - (ind[2] + ind[3]) + (ind[4] + ind[5]) - (ind[6] + ind[7])
    return value if value > 0.0 else 0.0,

# sets the fitness function, crossover settings, mutation rate, and the number of 
# fittest chromosomes that will be selected for the next generation
toolbox.register("evaluate", evaluateFitnessFunction)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb = 0.05)
toolbox.register("select", tools.selBest)

def main():
    textfile = open("chromosomes_max_fitness_output.txt", "w")
    
    random.seed(32)
    
    # 100 groups of random data
    pop = toolbox.population(n = 100)

    CXPB, MUTPB = 0.5, 0.2
    
    textfile.write("Start of Evolution\n")
    
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        
    textfile.write("  Evaluated %i chromosomes\n" %len(pop))
    
    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0
    
    # Begin the evolution, 10 generations
    while g < 10:
        # A new generation
        g = g + 1
        textfile.write("\n-- Generation %i --\n" % g)
        
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
    
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
    
        # Evaluate the individuals with an invalid fitness with a new fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        textfile.write("   Re-evaluated %i chromosomes\n" % len(invalid_ind))
        
        # The population is entirely replaced by the offspring
        textfile.write("   Chromosomes in the next generation %i\n" % len(offspring))
        pop[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        # Get the top 3 chromosomes
        topThree = tools.selBest(pop, 3)
        
        textfile.write("   Order of Most Fit Chromosome: %s\n" % fits)
        textfile.write("   Top 3 Chromosomes\n")
        textfile.write("   1. %s\n" % topThree[0])
        textfile.write("   2. %s\n" % topThree[1])
        textfile.write("   3. %s\n" % topThree[2])        
        textfile.write("   Max Fitness Value Output of population: %s\n" % max(fits))
    
    textfile.write("-- End of (successful) evolution --\n")
    
    best_ind = tools.selBest(pop, 1)[0]
    textfile.write("\nBest chromosome is %s, %s\n" % (best_ind, best_ind.fitness.values))

if __name__ == "__main__":
    main()
