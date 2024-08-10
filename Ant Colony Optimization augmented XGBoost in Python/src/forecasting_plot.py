import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import norm

class ForecastingPLot:
    def __init__(self, data, target_column, model_type='ARIMA'):
        self.data = data
        self.target_column = target_column
        self.model_type = model_type
        self.model = None

    def train_test_split(self, test_size):
        self.train, self.test = train_test_split(self.data, test_size=test_size, shuffle=False)
        #++

    def fit_model(self, order=(5, 1, 0), seasonal_order=(1, 1, 1, 12)):
        if self.model_type == 'ARIMA':
            self.model = ARIMA(self.train[self.target_column], order=order)
            self.model_fit = self.model.fit()

        elif self.model_type == 'XGBoost':
            self.X_train = np.array(range(len(self.train))).reshape(-1, 1)
            self.y_train = self.train[self.target_column].values
            self.model = XGBRegressor()
            self.model_fit = self.model.fit(self.X_train, self.y_train)

    def forecast(self, steps):
        if self.model_type == 'ARIMA':
            forecast_results = self.model_fit.get_forecast(steps=steps)
            self.forecast_values = forecast_results.predicted_mean
            self.conf_int = forecast_results.conf_int(alpha=0.05)

        elif self.model_type == 'XGBoost':
            X_test = np.array(range(len(self.train), len(self.train) + steps)).reshape(-1, 1)
            self.forecast_values = self.model_fit.predict(X_test)

            # Calculate confidence intervals based on residuals
            residuals = self.train[self.target_column].values - self.model_fit.predict(self.X_train)
            sigma = np.std(residuals)
            self.conf_int = pd.DataFrame({
                'lower Value': self.forecast_values - norm.ppf(0.975) * sigma,
                'upper Value': self.forecast_values + norm.ppf(0.975) * sigma
            })

    def plot_forecast(self, model_type):
        fig, ax = plt.subplots(figsize=(8, 4))
        plt.plot(self.data.index, self.data[self.target_column], label='Historical Data')

        plt.plot(pd.date_range(self.data.index[-1], periods=len(self.forecast_values)+1, freq='D')[1:],
                 self.forecast_values, label='Forecast', color='red')
        ax.axvline('01-01-2024', color='black', ls='--')

        plt.fill_between(pd.date_range(self.data.index[-1], periods=len(self.forecast_values)+1, freq='D')[1:],
                         self.conf_int.iloc[:, 0], self.conf_int.iloc[:, 1], color='pink', alpha=0.3, label='Confidence Interval')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'figs/{model_type}_forecasting.png')
        plt.show()


