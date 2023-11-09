'''
Гетун Дмитрий Юрьевич dmitry@getun.ru
Сссылка на учебный курс: https://online.mospolytech.ru/course/view.php?id=10055
Год разработки: 2023
'''

import csv
from random import randint
from itertools import combinations_with_replacement as combinate
from itertools import product
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

class Topology:
    
    def __init__(self):
        pass
    
    def generate_topology(self, path: str, size: tuple, bandwidth_range: tuple = (100, 1000)):
        rows, cols = size
        if rows != cols:
            raise ValueError("For a square matrix, the number of rows must be equal to the number of columns.")

        df = pd.DataFrame(index=range(1, rows + 1), columns=range(1, cols + 1))
        topology = []

        for i, j in product(range(1, rows + 1), repeat=2):
            if i < j:
                distance = randint(bandwidth_range[0], bandwidth_range[1])
                topology.append([i, j, distance])
                df.at[i, j] = distance
                df.at[j, i] = distance
            elif i == j:
                df.at[i, j] = 0

        df.to_csv(path, sep=';')
        #self.write_topology(path, topology)
        self.topology = df
        return df

    def write_topology(self, path, topology):
        fields = ['source', 'destination', 'distance']
        with open(path, 'w') as f:
            writer = csv.writer(f, delimiter=';', lineterminator='\n')
            writer.writerow(fields)
            writer.writerows(topology)

    def read_topology(self, path):
        df = pd.read_csv(path, sep=';', index_col=0)
        df.columns = list(map(int, df.columns))
        self.topology = df
        return df
            
        

    def step_generate_topology(self, path):
        size = int(input('Укажите размерность топологии'))
        paths = tuple(range(1, size + 1))
        combinations = list(combinate(paths, 2))
        topology = []
        for a, b in combinations:
            distance = int(input(f'Укажите пропускную способность для пути {a} -> {b}:'))
            t = (a, b, distance)
            topology.append(t)
        self.write_topology(path, topology)
        return topology

    def set_bound(self, start, stop):
        self.topology.at[start, stop] = 10000000000
        self.topology.at[stop, start] = 10000000000
        return self.topology
    

    def draw_graph(self):
        pass
    
    @property
    def matric(self):
        return self.topology
    
    
if __name__ == '__main__':
    t = Topology()
    r = t.read_topology('test_topology_min.csv')
    print(r)