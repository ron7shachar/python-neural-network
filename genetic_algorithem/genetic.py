from record.record import Record
import mnist_loader
from .NN_cricher import*
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()



def random_selection(population,fitness_fn):
    weights = [item.fitness for item in population]
    return random.choices(population,weights=weights,k=2)



def GENETIC_ALGORITHM(problem ,population,generations,alfa):
    # inputs: population, a set of individuals
    # FITNESS-FN, a function that measures the fitness of an individual

    record = Record("best_structure")
    record_ = record.get_record()
    if record_ is None:
        max_value = 0
    else:
        max_value = record_["fitness"]


    for i in range(generations):
        print(f"Generation ______  {i} ______")
        new_population=[]
        for i in range(len(population)):
            parent1,parent2 = random_selection(population,[cricher.fitness for cricher in population])
            child = parent1.reproduce(parent2)
            child.mutate()
            child.performing(problem)
            child.update_fitness()
            new_population.append(child)
        population = new_population
        best = max(population,key=lambda cricher:cricher.fitness)
        if max_value < best.fitness:
            max_value  = best.fitness
            my_object = {"hidden" : best.structure.hidden, "properties" : best.properties,"fitness":best.fitness}
            record.set_record(my_object)