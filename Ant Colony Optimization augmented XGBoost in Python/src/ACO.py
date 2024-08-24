import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


class AntColonyOptimizer:
    def __init__(self, n_ants, n_iterations, decay, alpha=1, beta=2):
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.pheromone = None
        self.best_path = None
        self.best_error = float('inf')
        self.fitted = False

    def fit(self, time_series, predict_steps):
        self.time_series = time_series
        self.predict_steps = predict_steps
        self.pheromone = np.ones(len(time_series))
        self.best_path, self.best_error = self._run_optimization()
        self.fitted = True

    def predict(self):
        if not self.fitted:
            raise ValueError("The model must be fitted before predicting.")
        return self.best_path

    def _run_optimization(self):
        # Initialize with a random path and its error
        initial_path = self._gen_path()
        best_path = initial_path
        best_error = self._evaluate_path(initial_path)

        for iteration in range(self.n_iterations):
            all_paths = self._gen_all_paths()
            for path, error in all_paths:
                if error < best_error:
                    best_path = path
                    best_error = error
            self._spread_pheromone(all_paths)
            self.pheromone *= self.decay

        return best_path, best_error

    def _gen_all_paths(self):
        all_paths = []
        for _ in range(self.n_ants):
            path = self._gen_path()
            error = self._evaluate_path(path)
            all_paths.append((path, error))
        return all_paths

    def _gen_path(self):
        path = []
        print(self.time_series)
        current_value = self.time_series[-1]

        for _ in range(self.predict_steps):
            move_probs = self._calculate_move_probabilities()
            next_value = self._choose_next_value(current_value, move_probs)
            path.append(next_value)
            current_value = next_value

        return path

    def _calculate_move_probabilities(self):
        move_probs = np.power(self.pheromone, self.alpha)
        return move_probs / np.sum(move_probs)

    def _choose_next_value(self, current_value, move_probs):
        possible_moves = np.array([-1, 1])  # For simplicity, moving down or up
        move_indices = np.arange(len(possible_moves))

        # Ensure the probabilities match the number of possible moves
        move_probs = move_probs[:len(possible_moves)]  # Match size
        move_probs /= move_probs.sum()  # Normalize to sum to 1

        # Choose the next move based on probabilities
        next_move = np.random.choice(move_indices, p=move_probs)

        # Return the updated current value
        return current_value + possible_moves[next_move]

    def _evaluate_path(self, path):
        predicted_series = np.concatenate([self.time_series, path])
        true_future = self._simulate_future()
        return mean_squared_error(true_future, predicted_series[-len(path):])

    def _spread_pheromone(self, all_paths):
        for path, error in all_paths:
            for idx in range(len(path)):
                self.pheromone[idx] += 1.0 / error

    def _simulate_future(self):
        # A simple simulation of the future, which could be replaced by actual future values if available
        return self.time_series[-self.predict_steps:] + np.random.normal(0, 0.1, self.predict_steps)
