import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import cross_val_score


class Ant:
    def __init__(self, param_space):
        self.param_space = param_space
        self.position = self.random_position()
        self.cost = None

    def random_position(self):
        return {key: np.random.choice(values) for key, values in self.param_space.items()}

    def evaluate(self, X, y):
        model = xgb.XGBRegressor(**self.position)
        scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=3)
        self.cost = -scores.mean()


class ACOOptimizer:
    def __init__(self, param_space, num_ants=10, num_generations=20, alpha=1, beta=2, evaporation_rate=0.5):
        self.param_space = param_space
        self.num_ants = num_ants
        self.num_generations = num_generations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.best_cost = np.inf
        self.best_position = None
        self.cost_history = []

        # Initialize pheromones
        self.pheromones = {key: np.ones(len(values)) for key, values in param_space.items()}

    def optimize(self, X, y):
        for generation in range(self.num_generations):
            ants = [Ant(self.param_space) for _ in range(self.num_ants)]

            for ant in ants:
                ant.evaluate(X, y)

                if ant.cost < self.best_cost:
                    self.best_cost = ant.cost
                    self.best_position = ant.position

            self.update_pheromones(ants)

            self.cost_history.append(self.best_cost)
            print(f"Generation {generation + 1}: Best Cost = {self.best_cost}")

    def update_pheromones(self, ants):
        for key in self.param_space.keys():
            for i in range(len(self.param_space[key])):
                self.pheromones[key][i] *= (1 - self.evaporation_rate)

            for ant in ants:
                idx = list(self.param_space[key]).index(ant.position[key])
                self.pheromones[key][idx] += 1 / ant.cost

    def plot_cost_vs_generation(self):
        plt.figure(figsize=(6, 6))
        plt.plot(range(1, self.num_generations + 1), self.cost_history, marker='o')
        plt.xlabel("Generation")
        plt.ylabel("Best Cost")
        plt.title("Best Cost vs Generation in ACO")
        plt.grid(False)
        plt.tight_layout()
        plt.savefig("figs/best_cost_vs_iteration.png", dpi=400)
        plt.show()
