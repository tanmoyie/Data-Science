import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
import matplotlib.dates as mdates
from src.ACO import AntColonyOptimizer
class ForecastingPLot:
    def __init__(self, X_train, X_test, y_train, y_test, target_column, model_type, forecast_horizon=13):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.target_column = target_column
        self.model_type = model_type
        self.forecast_horizon = forecast_horizon

    def train_test_split(self, test_size):
        pass
        #self.train, self.test = train_test_split(self.data, test_size=test_size, shuffle=False)
        #++

    def fit_model(self, order=(2, 0, 0), seasonal_order=(1, 1, 1, 12)):
        if self.model_type == 'ARIMA':
            self.model = ARIMA(self.y_train, order=order)
            self.model_fit = self.model.fit()

        elif self.model_type == 'XGBoost':
            self.model = XGBRegressor(n_estimators=500, max_depth=37, learning_rate=0.002)
            self.model_fit = self.model.fit(self.X_train, self.y_train)

        elif self.model_type == 'ACO':
            self.model = AntColonyOptimizer(n_ants=20, n_iterations=50, decay=0.95, alpha=1, beta=2)
            self.model_fit = self.model.fit(self.y_train, self.forecast_horizon)
            print(self.model_fit)

        elif self.model_type == 'ACO-XGBoost':
            self.model = XGBRegressor() # ++
            self.model_fit = self.model.fit(self.X_train, self.y_train)

    def forecast(self, steps):
        if self.model_type == 'ARIMA':
            forecast_results = self.model_fit.get_forecast(steps=steps)
            self.forecast_values = forecast_results.predicted_mean
            self.conf_int = forecast_results.conf_int(alpha=0.05)

        elif self.model_type == 'XGBoost':
            self.forecast_values = self.model_fit.predict(self.X_test)
            print(len(self.forecast_values))
            residuals = self.y_train - self.model_fit.predict(self.X_train)
            sigma = np.std(residuals)
            self.conf_int = pd.DataFrame({
                'lower Value': self.forecast_values - norm.ppf(0.975) * sigma,
                'upper Value': self.forecast_values + norm.ppf(0.975) * sigma
            })

        elif self.model_type == 'ACO':
            self.forecast_values = self.model_fit.predict()
            print(len(self.forecast_values))
            residuals = self.y_train - self.model_fit.predict()
            sigma = np.std(residuals)
            self.conf_int = pd.DataFrame({
                'lower Value': self.forecast_values - norm.ppf(0.975) * sigma,
                'upper Value': self.forecast_values + norm.ppf(0.975) * sigma
            })

        elif self.model_type == 'ACO-XGBoost':
            self.forecast_values = self.model_fit.predict(self.X_test)
            # Calculate confidence intervals based on residuals
            residuals = self.y_train - self.model_fit.predict(self.X_train)
            sigma = np.std(residuals)
            self.conf_int = pd.DataFrame({
                'lower Value': self.forecast_values - norm.ppf(0.975) * sigma,
                'upper Value': self.forecast_values + norm.ppf(0.975) * sigma
            })
            # print(self.conf_int)

    def plot_forecast(self, model_type, d):
        fig, ax = plt.subplots(figsize=(5, 3))
        plt.plot(self.X_train.index, self.y_train, label='Historical Data')

        ax.axvline(pd.to_datetime('01-01-2024'), color='black', ls='--')
        if model_type == 'ARIMA':
            plt.scatter(pd.date_range(self.X_test.index[0], periods=len(self.forecast_values), freq='D'),
                        self.forecast_values, label='ARIMA Forecast', marker='o', color='red', alpha=0.5)
            plt.fill_between(pd.date_range(self.X_test.index[0], periods=len(self.forecast_values), freq='D'),
                             self.conf_int[:, 0], self.conf_int[:, 1], color='pink', alpha=0.3, label='Confidence Interval')
        elif model_type == 'XGBoost':
            plt.scatter(pd.date_range(self.X_test.index[0], periods=len(self.forecast_values), freq='W'),
                     self.forecast_values, label='XGBoost Forecast', color='black', marker='s', alpha=0.5)
            plt.fill_between(pd.date_range(self.X_test.index[0], periods=len(self.forecast_values), freq='W'),
                             self.conf_int.iloc[:, 0], self.conf_int.iloc[:, 1], color='pink', alpha=0.3,
                             label='Confidence Interval')
        elif model_type == 'ACO':
            plt.plot(pd.date_range(self.X_train.index[-1], periods=len(self.forecast_values) + 1, freq='D')[1:],
                     self.forecast_values, label='ACO Forecast', color='black', marker='+', alpha=0.1)
            plt.fill_between(pd.date_range(self.X_train.index[-1], periods=len(self.forecast_values) + 1, freq='D')[1:],
                             self.conf_int.iloc[:, 0], self.conf_int.iloc[:, 1], color='pink', alpha=0.3,
                             label='Confidence Interval')
        elif model_type == 'ACO-XGBoost':
            plt.scatter(pd.date_range(self.X_test.index[0], periods=len(self.forecast_values), freq='W'),
                     self.forecast_values, label='ACO-XGBoost Forecast', color='blue', marker='o', alpha=0.5)
            plt.fill_between(pd.date_range(self.X_test.index[0], periods=len(self.forecast_values) , freq='W'),
                             self.conf_int.iloc[:, 0], self.conf_int.iloc[:, 1], color='red', alpha=0.5,
                             label='Confidence Interval')



        #ax.set_xticks()
        ax.set_xticklabels(d, rotation=75)
        plt.ylabel('Price')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(f'figs/{model_type}_forecasting.png', dpi=400)
        plt.show()


