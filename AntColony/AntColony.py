import numpy as np
import random
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt

class Salesman:
    def __init__(self, num_cities, x_lim, y_lim):
        self.num_cities = num_cities
        self.x_lim = x_lim
        self.y_lim = y_lim
        x_loc = np.random.uniform(0, x_lim, size=num_cities)
        y_loc = np.random.uniform(0, y_lim, size=num_cities)
        self.city_locations = [(x, y) for x, y in zip(x_loc, y_loc)]
        self.distances = self.calculate_distances()

    def calculate_distances(self):
        distances = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(i + 1, self.num_cities):
                dist = np.sqrt((self.city_locations[i][0] - self.city_locations[j][0]) ** 2 + (self.city_locations[i][1] - self.city_locations[j][1]) ** 2)
                distances[i][j] = distances[j][i] = dist
        return distances

    def fitness(self, solution):
        total_distance = 0
        for i in range(self.num_cities - 1):
            total_distance += self.distances[solution[i]][solution[i+1]]
        total_distance += self.distances[solution[-1]][solution[0]]
        fitness = -total_distance
        return fitness

class SimpleACO:
    def __init__(self, salesman, n_ants, rho):
        self.salesman = salesman
        self.n_ants = n_ants
        self.rho = rho
        self.pheromone = 0.01*np.ones((self.salesman.num_cities, self.salesman.num_cities))

    def update_pheromone(self, solutions):
        self.pheromone *= (1 - self.rho)
        for solution in solutions:
            for i in range(len(solution) - 1):
                self.pheromone[solution[i]][solution[i+1]] += 1 / -self.salesman.fitness(solution)

    def choose_next_city(self, current_city, path):
        probabilities = []
        for city in range(self.salesman.num_cities):
            if city not in path:
                probabilities.append(self.pheromone[current_city][city])
            else:
                probabilities.append(0)
        probabilities = np.array(probabilities)/np.sum(probabilities)
        next_city = np.random.choice(range(self.salesman.num_cities), p=probabilities)
        return next_city

    def generate_path(self):
        path = [random.randint(0, self.salesman.num_cities - 1)]
        while len(path) < self.salesman.num_cities:
            next_city = self.choose_next_city(path[-1], path)
            path.append(next_city)
        path.append(path[0])
        return path

    def run(self, n_iterations):
        best_distance = float('inf')
        best_solution = None
        for _ in tqdm(range(n_iterations)):
            solutions = [self.generate_path() for _ in range(self.n_ants)]
            self.update_pheromone(solutions)
            for solution in solutions:
                distance = -self.salesman.fitness(solution)
                if distance < best_distance:
                    best_distance = distance
                    best_solution = solution
        return best_solution

if __name__ == '__main__':
        # Instantiate the Salesman class with the number of cities and the limits for x and y coordinates
    salesman = Salesman(num_cities=10, x_lim=100, y_lim=100)

    # Instantiate the SimpleACO class with the Salesman instance, the number of ants, and the rho parameter
    aco = SimpleACO(salesman, n_ants=10, rho=0.5)

    # Run the ACO algorithm for a specified number of iterations
    n_iterations = 100
    best_solution = aco.run(n_iterations)

    # Print the best solution found
    print("Best solution found:", best_solution)