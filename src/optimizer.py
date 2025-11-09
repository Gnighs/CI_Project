import numpy as np
import cma
from scipy.optimize import minimize


class GeneticAlgorithm:
    def __init__(self, mlp, X, y, pop_size=20, mutation_rate=0.1,
                 mutation_scale=0.1, n_generations=50, n_elite=1):
        self.mlp = mlp
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.n_generations = n_generations
        self.n_elite = n_elite

        self.genome_length = mlp.n_params()
        self.population = [np.random.randn(self.genome_length)
                          for _ in range(pop_size)]
        
        # Tracking convergence
        self.history = {
            'generations': [],
            'best_fitness': [],
            'mean_fitness': [],
            'std_fitness': [],
            'diversity': []
        }

    def _evaluate_fitness(self, genome):
        mse = self.mlp.calculate_mse(self.X, self.y, genome)
        return -mse
    
    def _calculate_diversity(self):
        if len(self.population) < 2:
            return 0.0
        
        # Calculate average pairwise Euclidean distance
        distances = []
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                dist = np.linalg.norm(self.population[i] - self.population[j])
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0

    def _select_parents(self, fitnesses):
        shifted = fitnesses - np.min(fitnesses) + 1e-6
        probs = shifted / np.sum(shifted)
        idx = np.random.choice(len(self.population), size=2, p=probs, replace=False)
        return self.population[idx[0]], self.population[idx[1]]

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
            fitnesses = np.array([self._evaluate_fitness(ind)
                                 for ind in self.population])
            
            # Track convergence metrics
            self.history['generations'].append(gen + 1)
            self.history['best_fitness'].append(np.max(fitnesses))
            self.history['mean_fitness'].append(np.mean(fitnesses))
            self.history['std_fitness'].append(np.std(fitnesses))
            self.history['diversity'].append(self._calculate_diversity())

            # Keep best
            elite_indices = np.argsort(fitnesses)[-self.n_elite:]
            elites = [self.population[i].copy() for i in elite_indices]
            new_population = elites.copy()

            while len(new_population) < self.pop_size:
                p1, p2 = self._select_parents(fitnesses)
                c1, c2 = self._crossover(p1, p2)
                new_population.append(self._mutate(c1))
                if len(new_population) < self.pop_size:
                    new_population.append(self._mutate(c2))

            self.population = new_population
            if (gen + 1) % 10 == 0:
                print(f"Generation {gen+1}: "
                      f"best fitness = {np.max(fitnesses):.5f}")

        fitnesses = np.array([self._evaluate_fitness(ind)
                              for ind in self.population])
        best_idx = np.argmax(fitnesses)
        return self.population[best_idx], self.history


class CMAES:
    def __init__(self, mlp, X, y, pop_size=20, n_generations=50, sigma0=0.5):
        self.mlp = mlp
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.sigma0 = sigma0
        self.genome_length = mlp.n_params()
        
        # Tracking convergence
        self.history = {
            'generations': [],
            'best_fitness': [],
            'mean_fitness': [],
            'std_fitness': [],
            'diversity': []
        }
        
    def _evaluate_fitness(self, genome):
        mse = self.mlp.calculate_mse(self.X, self.y, genome)
        return mse
    
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
            
            # Track convergence metrics
            self.history['generations'].append(generation + 1)
            self.history['best_fitness'].append(-es.result.fbest)
            self.history['mean_fitness'].append(-np.mean(fitness_values))
            self.history['std_fitness'].append(np.std(fitness_values))
            
            # Calculate diversity
            if len(solutions) > 1:
                distances = []
                for i in range(len(solutions)):
                    for j in range(i + 1, len(solutions)):
                        dist = np.linalg.norm(solutions[i] - solutions[j])
                        distances.append(dist)
                diversity = np.mean(distances) if distances else 0.0
            else:
                diversity = 0.0
            self.history['diversity'].append(diversity)
            
            generation += 1
            if generation % 10 == 0:
                print(f"Generation {generation}: "
                      f"best fitness = {-es.result.fbest:.5f}")
        
        return es.result.xbest, self.history


class LBFGSB:
    def __init__(self, mlp, X, y, maxiter=500):
        self.mlp = mlp
        self.X = X
        self.y = y
        self.maxiter = maxiter
        
        # Tracking convergence
        self.history = {
            'iterations': [],
            'fitness': []
        }
    
    def _evaluate_fitness(self, genome):
        mse = self.mlp.calculate_mse(self.X, self.y, genome)
        return mse
    
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
