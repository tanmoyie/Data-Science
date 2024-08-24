import matplotlib.pyplot as plt
import numpy as np


class TimeSeriesPlotter:
    def __init__(self, time, actual, predicted, lower_ci, upper_ci, model_name):
        self.time = time
        self.actual = actual
        self.predicted = predicted
        self.lower_ci = lower_ci
        self.upper_ci = upper_ci
        self.model_name = model_name

    def plot(self, model_name):
        n_plots = len(self.actual)
        fig, axes = plt.subplots(n_plots, 1, figsize=(6, 3 * n_plots))

        if n_plots == 1:
            axes = [axes]  # Ensure axes is always a list

        for i, ax in enumerate(axes):
            ax.plot(self.time[i], self.actual[i], label='Actual', color='gray', linestyle='--')
            if model_name is 'ACO-XGBoost':
                ax.plot(self.time[i], self.predicted[i], label='Predicted', color='blue')
            else:
                ax.plot(self.time[i], self.predicted[i], label='Predicted', color='black')
            ax.fill_between(self.time[i], self.lower_ci[i], self.upper_ci[i], color='gray', alpha=0.2,
                            label='Confidence Interval')

            ax.set_xlabel('Time')
            ax.set_ylabel('Fuel Price')
            ax.legend(loc='upper left', fontsize=11)


        plt.tight_layout()
        plt.savefig(f'figs/{model_name} times series plot C.I.png', dpi=500)
        plt.show()
