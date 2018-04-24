# -*- coding: utf-8 -*-
import random
import numpy

from deap import base
from deap import creator
from deap import tools

# create the fitness base and weight
# individual specifies the groups of data input representing each 
# group of chromosomes in array of array
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMax)

# specifies the toolbox and the population data generation settings
# populates groups with random int data between 0 and 9
toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 0, 9)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, 100)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

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
toolbox.register("select", tools.selTournament, tournsize = 3)

def main():
    textfile = open("chromosomes_max_fitness_output.txt", "w")
    
    random.seed(64)
    
    # 100 groups of random data
    pop = toolbox.population(n = 8)

    CXPB, MUTPB = 0.5, 0.2
    
    textfile.write("Start of Evolution\n")
    
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        
    textfile.write("  Evaluated %i individuals\n" %len(pop))
    
    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0
    
    # Begin the evolution, 25 generations
    while max(fits) < 100 and g < 25:
        # A new generation
        g = g + 1
        textfile.write("-- Generation %i --\n" % g)
        
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
    
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        textfile.write("  Evaluated %i individuals\n" % len(invalid_ind))
        
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        fitnessVal = evaluateFitnessFunction(fits)
        
        textfile.write("   Individual Group: %s\n" % fits)
        textfile.write("   Fitness Value Output: %s\n" % fitnessVal)
    
    textfile.write("-- End of (successful) evolution --\n")
    
    best_ind = tools.selBest(pop, 1)[0]
    textfile.write("Best individual is %s, %s\n" % (best_ind, best_ind.fitness.values))

if __name__ == "__main__":
    main()
