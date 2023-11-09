from utils import Individual
import random
import numpy as np

class Selection:
    
    @classmethod
    def __all_estimate(cls, population):
        return sum([individual.estimate for individual in population])
    
    @classmethod
    def tournament(cls, population):
        group_size = 2
        new_population = []
        while len(new_population) != population:
            random.shuffle(population)
            groups = np.array_split(population)
        
    
    @classmethod
    def wheel(cls, population: list[Individual]):
        full_estimate = cls.__all_estimate(population)
        for individual in population:
            individual.percent = (individual.estimate / full_estimate) * 100
        
        
        new_population = random.choices(
            population, 
            weights=[individual.percent for individual in population],
            k=len(population)
        )
        
        return new_population
    
    def rang_selection(cls, population: list[Individual]):
        full_estimate = cls.__all_estimate(population)
        for individual in population:
            individual.percent = (individual.estimate / full_estimate) * 100
        
        new_population: list[Individual] = sorted(individual, lambda x: x.percent)
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
    
class Crossingover:
    
    @classmethod
    def random_parents(cls, population):
        family = []
        for _ in range(len(population)):
            parents = random.choices(population, k=2)
            family.append(parents)
        return family
    
    
    
    @classmethod
    def one_point(cls, population):
        pass
    
    @classmethod
    def two_point(cls, population):
        pass
    
    @classmethod
    def homo_cross(cls, population):
        pass
    
    
class Mutation:
    
    @classmethod
    def invert_vortex(cls, individual):
        pass
    
    @classmethod
    def permutation(cls, individual):
        pass