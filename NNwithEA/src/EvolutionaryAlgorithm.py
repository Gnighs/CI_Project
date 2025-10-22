import numpy as np

class EvolutionaryAlgorithm:
    def __init__(self, mlp, X, y, pop_size=20, mutation_rate=0.1, mutation_scale=0.1, n_generations=50, n_elite=1):
        self.mlp = mlp
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.n_generations = n_generations
        self.n_elite = n_elite

        self.genome_length = mlp.n_params()
        self.population = [np.random.randn(self.genome_length) for _ in range(pop_size)]

    def _evaluate_fitness(self, genome):
        mse = self.mlp.calculate_mse(self.X, self.y, genome)
        return -mse

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
            fitnesses = np.array([self._evaluate_fitness(ind) for ind in self.population])

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
            print(f"Generation {gen+1}: best fitness = {np.max(fitnesses):.5f}")

        fitnesses = np.array([self._evaluate_fitness(ind) for ind in self.population])
        best_idx = np.argmax(fitnesses)
        return self.population[best_idx]
