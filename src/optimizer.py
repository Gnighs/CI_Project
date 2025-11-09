import numpy as np
import cma
from scipy.optimize import minimize
from abc import ABC, abstractmethod
import random

class BaseClass(ABC):
    def __init__(self, mlp, X, y):
        self.mlp = mlp
        self.X = X
        self.y = y
        self.history = {}


    def _evaluate_fitness(self, genome):
        mse = self.mlp.calculate_mse(self.X, self.y, genome)
        return mse


    def _calculate_diversity(self, population):
        if len(population) < 2:
            return 0.0
        
        # Calculate average pairwise Euclidean distance
        distances = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                dist = np.linalg.norm(population[i] - population[j])
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0


    @abstractmethod
    def run(self):
        pass



class EvolutionaryAlgorithm(BaseClass):
    def __init__(self, mlp, X, y, pop_size=20, mutation_rate=0.1,
                 mutation_scale=0.1, n_generations=50, n_elite=5,
                 selection_method = 'roulette'):
        super().__init__(mlp, X, y)
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.n_generations = n_generations
        self.n_elite = n_elite
        self.selection_method = selection_method

        self.genome_length = mlp.n_params()
        self.population = [np.random.randn(self.genome_length)
                           for _ in range(pop_size)]

        self.history = {
            'generations': [],
            'best_fitness': [],
            'mean_fitness': [],
            'std_fitness': [],
            'diversity': []
        }

    def _roulette_selection(self, fitnesses):
        shifted = fitnesses - np.min(fitnesses) + 1e-6
        probs = shifted / np.sum(shifted)
        idx = np.random.choice(len(self.population), size=2, p=probs, replace=False)
        return self.population[idx[0]], self.population[idx[1]]


    def _tournament_selection(self, fitnesses, k=4):
        idx = np.random.choice(len(self.population), size=k, replace=False)
        selected = sorted(idx, key=lambda i: fitnesses[i], reverse=True)
        return self.population[selected[0]], self.population[selected[1]]


    def _select_parents(self, fitnesses):
        if self.selection_method == 'roulette':
            return self._roulette_selection(fitnesses)
        elif self.selection_method == 'tournament':
            return self._tournament_selection(fitnesses, k=3)


    def _crossover(self, parent1, parent2):
        point = np.random.randint(1, self.genome_length)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2


    def _mutate(self, genome):
        for i in range(self.genome_length):
            if np.random.rand() < self.mutation_rate:
                genome[i] += np.random.randn() * self.mutation_scale
        return genome


    def run(self):
        for gen in range(self.n_generations):

            fitnesses = np.array([-self._evaluate_fitness(ind) for ind in self.population])
            self.history['generations'].append(gen + 1)
            self.history['best_fitness'].append(np.max(fitnesses))
            self.history['mean_fitness'].append(np.mean(fitnesses))
            self.history['std_fitness'].append(np.std(fitnesses))
            self.history['diversity'].append(self._calculate_diversity(self.population))


            # Selection phase, roulette or tournament

            parent_pool = []

            for _ in range(self.pop_size // 2): # Each pair produces two offspring
                p1, p2 = self._select_parents(fitnesses)
                parent_pool.append((p1, p2))


            # Reproduction phase
            offspring = []
            for (p1, p2) in parent_pool:
                c1, c2 = self._crossover(p1, p2)
                offspring.append(self._mutate(c1))
                offspring.append(self._mutate(c2))

            # Replacement phase

            elite_indices = np.argsort(fitnesses)[-self.n_elite:]
            elites = [self.population[i].copy() for i in elite_indices]

            random.shuffle(offspring)

            self.population = elites + offspring[:self.pop_size - self.n_elite]

        best_idx = np.argmax(fitnesses)
        return self.population[best_idx], self.history


class CMAES(BaseClass):
    def __init__(self, mlp, X, y, pop_size=20, n_generations=50, sigma0=0.5):
        super().__init__(mlp, X, y)
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.sigma0 = sigma0
        self.genome_length = mlp.n_params()

        self.history = {
            'generations': [],
            'best_fitness': [],
            'mean_fitness': [],
            'std_fitness': [],
            'diversity': []
        }

    def run(self):
        initial_mean = np.random.randn(self.genome_length) * 0.1

        opts = {
            'popsize': self.pop_size, 
            'maxiter': self.n_generations, 
            'verb_disp': 0, 
            'verbose': -9
        }
        es = cma.CMAEvolutionStrategy(initial_mean, self.sigma0, opts)

        generation = 0
        while not es.stop() and generation < self.n_generations:
            solutions = es.ask()
            fitness_values = [self._evaluate_fitness(x) for x in solutions]
            es.tell(solutions, fitness_values)

            self.history['generations'].append(generation + 1)
            self.history['best_fitness'].append(-es.result.fbest)
            self.history['mean_fitness'].append(-np.mean(fitness_values))
            self.history['std_fitness'].append(np.std(fitness_values))
            self.history['diversity'].append(self._calculate_diversity(solutions))

            generation += 1

        return es.result.xbest, self.history


class LBFGSB(BaseClass):
    def __init__(self, mlp, X, y, maxiter=500):
        super().__init__(mlp, X, y)

        self.maxiter = maxiter

        self.history = {
            'iterations': [], 
            'fitness': []
        }

    def run(self):
        initial_genome = np.random.randn(self.mlp.n_params())
        iteration = [0]

        def objective(genome):
            mse = self._evaluate_fitness(genome)
            iteration[0] += 1
            self.history['iterations'].append(iteration[0])
            self.history['fitness'].append(-mse)
            return mse

        result = minimize(objective, initial_genome, method='L-BFGS-B',
                          options={'maxiter': self.maxiter})
        return result.x, self.history

