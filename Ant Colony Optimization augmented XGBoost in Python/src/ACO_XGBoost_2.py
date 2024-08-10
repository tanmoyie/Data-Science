import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Step 1: Data Preparation
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


# Step 2: Feature Construction (Example)
def create_features(data):
    # Example feature: lagged values
    for lag in range(1, 6):
        data[f'lag_{lag}'] = data['price'].shift(lag)
    data.dropna(inplace=True)
    return data


# Step 3: ACO Initialization
class ACO:
    def __init__(self, n_ants, n_iterations, n_features, evaporation_rate, pheromone_deposit_rate):
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.n_features = n_features
        self.evaporation_rate = evaporation_rate
        self.pheromone_deposit_rate = pheromone_deposit_rate
        self.pheromone = np.ones(n_features)

    def _select_features(self):
        features = []
        for i in range(self.n_features):
            if np.random.rand() < self.pheromone[i]:
                features.append(i)
        return features

    def optimize(self, X, y):
        best_score = float('inf')
        best_features = []

        for _ in range(self.n_iterations):
            all_scores = []
            all_features = []

            for _ in range(self.n_ants):
                selected_features = self._select_features()
                if len(selected_features) == 0:
                    continue
                X_selected = X[:, selected_features]
                score = self._evaluate(X_selected, y)
                all_scores.append(score)
                all_features.append(selected_features)

                if score < best_score:
                    best_score = score
                    best_features = selected_features

            self._update_pheromone(all_features, all_scores)

        return best_features, best_score

    def _evaluate(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = xgb.XGBRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = mean_squared_error(y_test, y_pred)
        return score

    def _update_pheromone(self, all_features, all_scores):
        for i in range(self.n_features):
            self.pheromone[i] *= (1 - self.evaporation_rate)

        for features, score in zip(all_features, all_scores):
            for feature in features:
                self.pheromone[feature] += self.pheromone_deposit_rate / score

