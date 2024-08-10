import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# Sample time series data
def generate_time_series(n=100):
    np.random.seed(42)
    return pd.Series(np.sin(np.linspace(0, 10, n)) + np.random.normal(0, 0.5, n))

# Objective function: calculate the error of the forecasted values
def objective_function(predicted, actual):
    return mean_squared_error(actual, predicted)

# ACO parameters
n_ants = 10
n_iterations = 50
n_future = 10  # Number of future points to forecast
alpha = 1.0  # Pheromone importance
beta = 2.0  # Heuristic importance
rho = 0.5  # Evaporation rate

# Generate sample time series
data = generate_time_series()

# Initialize pheromones
pheromones = np.ones((n_future, len(data)))

# Initialize heuristic (e.g., based on recent trends)
def heuristic(data, current_value):
    return np.exp(-np.abs(data - current_value))

# ACO main loop
best_forecast = None
best_error = np.inf

for iteration in range(n_iterations):
    all_forecasts = []
    all_errors = []

    for ant in range(n_ants):
        forecast = []
        current_value = data.iloc[-1]  # Start from the last value

        for t in range(n_future):
            # Calculate probabilities for moving to each possible future value
            heuristic_values = heuristic(data, current_value)
            probs = (pheromones[t, :] ** alpha) * (heuristic_values ** beta)
            probs /= np.sum(probs)

            # Move to the next value based on probabilities
            next_value_index = np.random.choice(len(data), p=probs)
            next_value = data.iloc[next_value_index]

            # Update forecast
            forecast.append(next_value)
            current_value = next_value

        # Calculate error with respect to the actual future data
        actual = data[-n_future:]
        error = objective_function(forecast, actual)

        all_forecasts.append(forecast)
        all_errors.append(error)

        # Update best forecast if this one is better
        if error < best_error:
            best_error = error
            best_forecast = forecast

    # Update pheromones
    for t in range(n_future):
        for ant in range(n_ants):
            pheromones[t, :] *= (1 - rho)
            pheromones[t, :] += rho / (1 + all_errors[ant])

# Output the best forecast
print("Best Forecast:", best_forecast)
print("Best Error:", best_error)
