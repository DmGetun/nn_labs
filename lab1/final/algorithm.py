'''
Гетун Дмитрий Юрьевич dmitry@getun.ru
Сссылка на учебный курс: https://online.mospolytech.ru/course/view.php?id=10055
Год разработки: 2023
'''
from topology import Topology
from individual import Individual
import random
import itertools as it
import numpy as np
import time

class Algorithm(list):
    def __init__(
            self,
            pop_size=10, 
            step_count=300, 
            ind_length = 10,
            mode = 'full',
            graph = None,
            mutation_rate = 0.1,
            elite_size = 3
            ):
        self.pop_size = pop_size
        self.step_count = step_count
        self.ind_length = ind_length
        self.mode = mode
        self.graph = graph
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size

    def read_graph(self, path):
        self.graph = Topology().read_topology(path)
        self.pop_size = len(self.graph.columns)
        return self.graph

    def set_bound(self, start, end):
        self.start_vertex = start
        self.end_vertex = end

    def init_population(self):
        population = []
        for _ in range(self.pop_size):
            individual = Individual(list(range(1, len(self.graph) + 1)))
            random.shuffle(individual)
            individual.direct_mutation(self.start_vertex, self.end_vertex)
            population.append(individual)
        return population
        
    def pairwise(self, iterable):
        a, b = it.tee(iterable)
        next(b, None)
        return zip(a, b)


    def evaluate(self, individual: Individual):
        roads = it.pairwise(individual)
        #lengths = [self.topology.matric.at[a,b] for a,b in roads]
        lengths = [self.graph.at[a,b] for a,b in roads]
        individual.estimate = sum(lengths)
        return individual

    def one_point_crossover(self, parent1, parent2):
        '''
            Одноточечное скрещивание
        '''
        idx = random.randint(1, len(parent1) - 1)
        child1 = Individual(parent1[:idx] + parent2[idx:])
        child1.direct_mutation(self.start_vertex, self.end_vertex)
        child2 = Individual(parent2[:idx] + parent1[idx:])
        child2.direct_mutation(self.start_vertex, self.end_vertex)
        return child1, child2
    
    def direct_crossover(self, parent1, parent2):
        childs = [Individual([]), Individual([])]
        childs[0].append(self.start_vertex)
        childs[1].append(self.start_vertex)

        for i in range(len(childs)):
            for gene1, gene2 in zip(parent1[1:-1], parent2[1:-1]):
                if random.random() < 0.5:
                    childs[i].append(gene1)
                else:
                    childs[i].append(gene2)

        childs[0].append(self.end_vertex)
        childs[1].append(self.end_vertex)
        return childs


    def swap_mutate(self, individual):
        '''
            Мутация обменом вершинами
        '''
        for i in range(1, len(individual) - 1):  # Исключим начальную и конечную вершины
            if random.random() < self.mutation_rate:
                j = random.randint(1, len(individual) - 2)
                individual[i], individual[j] = individual[j], individual[i]
        return individual
    
    def random_mutate(self, individual):
        for i in range(1, len(individual) - 1):  # Исключим начальную и конечную вершины
            if random.random() < self.mutation_rate:
                j = random.randint(1, len(individual) - 2)
                individual[i] = j
        return individual
    
    def roulette_selection(self, population):
        '''
        Селекция методом рулетки
        '''
        parents = []
        total_fitness = sum(1 / (1 + ind.estimate) for ind in population)
        normalized_scores = [(1 / (1 + ind.estimate)) / total_fitness for ind in population]

        cumulative_sum = [sum(normalized_scores[:i+1]) for i in range(len(normalized_scores))]

        random_point = random.random()
        for _ in range(len(population) // 2 - self.elite_size):
            for i in range(len(cumulative_sum)):
                if random_point <= cumulative_sum[i]:
                    parents.append(population[i])

        return parents

    def tournament_selection(self, population, tournament_size=3):
        parents = []
        random.shuffle(population)
        for i in range(0, self.pop_size, tournament_size):
            #selected = random.sample(population, tournament_size)
            selected = population[i: i + tournament_size]
            parents.append(min(selected, key=lambda x: x.estimate))
        return parents


    def fit(self, mode, crossover_variant=None, selection_variant=None, mutation_variant=None):
        population = self.init_population()

        for generation in range(self.step_count):
            scored_population = [self.evaluate(individ) for individ in population]
            scored_population.sort(key=lambda x: x.estimate)

            elite_individuals = scored_population[:self.elite_size]
            
            parents = selection_variant(scored_population)
            new_population = []
            while len(new_population) < self.pop_size - self.elite_size:
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)
                child1, child2 = crossover_variant(parent1, parent2)
                new_population += [child1, child2]

            new_population = [
                mutation_variant(ind) for ind in new_population
            ]
            
            means = [ind.estimate for ind in scored_population]

            population = elite_individuals + new_population
            best_individual = scored_population[0]
            print(f'Generation: {generation}, Best: {best_individual}, '\
                  f'Score: {best_individual.estimate}, Mean: {np.mean(means)}')

def path(df, start, end):

    # Удаление строки и столбца для начальной и конечной вершин, чтобы оставить только промежуточные вершины
    df_temp = df.drop([start, end], axis=0).drop([start, end], axis=1)

    # Вычисление суммы расстояний для каждого возможного пути
    min_distance = float('inf')
    best_vertex = None

    for vertex in df_temp.index:
        distance = df.at[start, vertex] + df.at[vertex, end]
        if distance < min_distance:
            min_distance = distance
            best_vertex = vertex

    # Результат
    print(f"The shortest path is {start} -> {best_vertex} -> {end} with distance {min_distance}")


if __name__ == '__main__':
    algorithm = Algorithm(
    pop_size=30,
    step_count=200,
    ind_length=10,
    mode='full',
    graph=None,
    mutation_rate=0.2
)
    random.seed(15)
    Topology().generate_topology('test123.csv', size=(10, 10))
    graph = algorithm.read_graph('test123.csv')
    print(graph)
    algorithm.set_bound(1, 10)
    
    crossovers = [algorithm.one_point_crossover, algorithm.direct_crossover]
    selections = [algorithm.roulette_selection, algorithm.tournament_selection]
    mutations = [algorithm.swap_mutate, algorithm.random_mutate]
    algorithm.fit(
        mode='full',
        crossover_variant=algorithm.direct_crossover,
        selection_variant=algorithm.tournament_selection,
        mutation_variant=algorithm.random_mutate
    )

    path(graph, 1, 10)
    time.sleep(1)
    if True:
        for crossover in crossovers:
            for selection in selections:
                for mutation in mutations:
                    algorithm.fit(
                        mode='full', 
                        crossover_variant=crossover,
                        selection_variant=selection,
                        mutation_variant=mutation
                    )
                    print(f'{crossover}, {selection}, {mutation}')
                    time.sleep(10)
            


    


    