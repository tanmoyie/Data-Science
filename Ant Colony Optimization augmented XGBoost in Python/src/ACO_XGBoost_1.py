import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error
import random


class AntColonyOptimizer:
    def __init__(self, num_ants, num_iterations, evaporation_rate, pheromone_deposition_rate, hyperparameter_ranges):
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.evaporation_rate = evaporation_rate
        self.pheromone_deposition_rate = pheromone_deposition_rate
        self.hyperparameter_ranges = hyperparameter_ranges
        self.pheromones = {param: np.ones(num_ants) for param in hyperparameter_ranges}
        self.best_params = None
        self.best_score = float('inf')

    def objective_function(self, params, X_train, y_train, X_test, y_test):
        model = XGBRegressor(
            n_estimators=int(params['n_estimators']),
            max_depth=int(params['max_depth']),
            learning_rate=params['learning_rate'],
            n_jobs=8  # ++
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return root_mean_squared_error(y_test, y_pred)

    def optimize(self, X_train, y_train, X_test, y_test):
        for iteration in range(self.num_iterations):
            ants = self._generate_ants()
            self._evaluate_ants(ants, X_train, y_train, X_test, y_test)
            self._update_pheromones(ants)

        print('Best Parameters:', self.best_params)
        print('Best RMSE Score:', self.best_score)
        print('--')

        return self.best_params, self.best_score

    def _generate_ants(self):
        ants = []
        for _ in range(self.num_ants):
            params = {}
            for param, (low, high) in self.hyperparameter_ranges.items():
                value = np.random.choice(np.linspace(low, high, self.num_ants))
                params[param] = value
            ants.append(params)
            print('ants', ants)
        return ants

    def _evaluate_ants(self, ants, X_train, y_train, X_test, y_test):
        for ant in ants:
            score = self.objective_function(ant, X_train, y_train, X_test, y_test)
            if score < self.best_score:
                self.best_score = score
                self.best_params = ant

    def _update_pheromones(self, ants):
        for param in self.pheromones:
            pheromone_update = np.array([ant[param] for ant in ants])
            self.pheromones[param] = (1 - self.evaporation_rate) * self.pheromones[param] + \
                                     self.pheromone_deposition_rate * pheromone_update

    def train_final_model(self, X_train, y_train):
        model = XGBRegressor(
            n_estimators=int(self.best_params['n_estimators']),
            max_depth=int(self.best_params['max_depth']),
            learning_rate=self.best_params['learning_rate']
        )
        model.fit(X_train, y_train)
        return model

        # is this code producing multiple hyperparam combinations??  ++
