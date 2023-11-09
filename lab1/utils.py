import itertools as it
import pandas as pd
import random
from topology import Topology
import numpy as np

class Individual(list):
    
    def __init__(self, *args):
        super().__init__(*args)
        
    @property
    def estimate(self):
        return self._estimate
    
    @estimate.setter
    def estimate(self, estimate):
        self._estimate = estimate
    
    @property
    def percent(self):
        return self._percent
    
    @percent.setter
    def percent(self, percent):
        self._percent = percent
        
    @property
    def order(self):
        return self._order
    
    @order.setter
    def order(self, order):
        self._order = order
        
    @property
    def rank(self):
        return self._rank
    
    @rank.setter
    def rank(self,rank):
        self._rank = rank
        
    @property
    def percent_rank(self):
        return self._percent_rank
    
    @percent_rank.setter
    def percent_rank(self, percent_rank):
        self._percent_rank = percent_rank
        
    def direct_mutation(self, start, stop):
            min = self.index(start)
            self[0] , self[min] = self[min] , self[0]
                        
            max = self.index(stop)
            self[-1] , self[max] = self[max] , self[-1]
        
    def permutation(self):
        vertexes = random.choices(range(1, len(self) - 1), k=2)
        self[vertexes[0]], self[vertexes[1]] = self[vertexes[1]], self[vertexes[0]]
        
    def mutation(self, start, stop):
        self.permutation()
        self.direct_mutation(start, stop)
        
        
class Algorithm:
    def __init__(
        self,
        size: int = 100,
        mutation_chance: float = 0.1,
        step_number: int = 10   
    ):
        self.size = size
        self.population: list[Individual] = []
        self.topology: Topology = []
        self.mutation_chance: float = mutation_chance
        self.step_number = step_number
        
    def read_topology(self, path):
        self.topology = Topology()
        self.topology.read_topology(path)
        return self.topology
    
    def set_bound(self, start, stop):
        self.topology.set_bound(start, stop)
        self.start = start
        self.stop = stop
        
        return self.topology
    
    def repair_bound(self):
        for individual in self.population:
            individual.direct_mutation(self.start,self.stop)
            
        return self.population
            

    def generate_population(self):
        '''
        Генерация начальной популяции
        '''
        for i in range(self.size):
            individual = Individual(list(range(1, len(self.topology.matric) + 1)))
            individual.order = i + 1
            random.shuffle(individual)
            self.population.append(individual)
        return self.population
        
    def min_func(self, individual: Individual):
        roads = self.pairwise(individual)
        lengths = [self.topology.matric.at[a,b] for a,b in roads]
        individual.way = sum(lengths)
        return sum(lengths)
    
    
    def pairwise(self, iterable):
        a, b = it.tee(iterable)
        next(b, None)
        return zip(a, b)
    
    def mean(self):
        return 1 / np.mean([individual.estimate for individual in self.population])
    
    def best_way(self):
        return np.min([individual.way for individual in self.population])
    
    def estimate(self):
        for individual in self.population:
            individual.estimate = 1 / (1 + self.min_func(individual))
        return self.population
    
    def all_estimate(self):
        return sum([individual.estimate for individual in self.population])
    
    def get_child(self, indexes, parents):
        child_1 = Individual([0] * len(parents[0]))
        child_2 = Individual([0] * len(parents[0]))

        start, stop = indexes[0], indexes[1]
        parent_1, parent_2 = parents[0], parents[1]

        child_1[start:stop] = parent_2[start:stop]
        child_2[start:stop] = parent_1[start:stop]

        insert_index = 0
        for i in range(len(parent_2)):
            vertex = parent_2[i]
            if (vertex not in child_2):
                child_2[insert_index] = vertex
                insert_index += 1
                if insert_index == start: insert_index = stop
                
        insert_index = 0
        for i in range(len(parent_1)):
            vertex = parent_1[i]
            if (vertex not in child_1):
                child_1[insert_index] = vertex
                insert_index += 1
                if insert_index == start: insert_index = stop

        return child_1, child_2
            
        
    def crossing(self):
        '''
        Скрещивание
        '''
        new_population = []
        while len(new_population) != 100:
            point_indexes = sorted(random.choices(range(1,len(self.population)),k=2))
            individuals = random.choices(self.population, k=2)
            childs = self.get_child(point_indexes, individuals)
            child_1 = childs[0]
            child_2 = childs[1]
            child_1.direct_mutation(self.start, self.stop)
            child_2.direct_mutation(self.start, self.stop)
            new_population.extend([child_1, child_2])
        self.population = new_population
        
    def rang_selection(self):
        full_estimate = self.all_estimate()
        for individual in self.population:
            individual.percent = (individual.estimate / full_estimate) * 100
        
        new_population: list[Individual] = sorted(self.population, key=lambda x: x.percent)
        for i, individual in enumerate(new_population):
            individual.rank = i + 1
            
        full_rank = sum([individual.rank for individual in new_population])
        
        for individual in new_population:
            individual.percent_rank = (individual.rank / full_rank) * 100
        
        new_population = random.choices(
            new_population, 
            weights=[individual.percent_rank for individual in new_population],
            k=len(new_population)
        )
        
        return new_population
    
    def selection(self):
        '''
        Селекция методом рулетки
        '''
        full_estimate = self.all_estimate()
        for individual in self.population:
            individual.percent = (individual.estimate / full_estimate) * 100
            
        wheel = list(it.accumulate([individual.percent for individual in self.population]))[:-1]
        wheel.insert(0, 0)
        
        self.population = random.choices(
            self.population, 
            weights=[individual.percent for individual in self.population],
            k=len(self.population)
        )
        
        return self.population
    
    def get_elite(self, count:int = 1):
        population = sorted(self.population, key=lambda x: x.estimate)
        return population[:count]
        
    
    def mutation(self):
        '''
        Мутация
        '''
        for individual in self.population:
            random_number = random.random()
            if random_number < self.mutation_chance:
                individual.mutation(self.start, self.stop)
    
    def fit(self):
        self.generate_population() # генерация начальной популяции
        for i in range(self.step_number):
            self.estimate() # оценка приспособленности особей
            self.selection() # селекция
            self.crossing() # скрещивание
            self.mutation() # мутация
            self.estimate()
            print(self.mean())
            
    
    
if __name__ == '__main__':
    path = 'test_topology.csv'

    algorithm = Algorithm(size=100, step_number=300)
    algorithm.read_topology(path)
    algorithm.set_bound(1, 4)
    alg = algorithm.fit()

        
        
    