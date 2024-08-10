import matplotlib.pyplot as plt
import numpy as np


class TimeSeriesPlotter:
    def __init__(self, time, actual, predicted, lower_ci, upper_ci):
        self.time = time
        self.actual = actual
        self.predicted = predicted
        self.lower_ci = lower_ci
        self.upper_ci = upper_ci

    def plot(self, title_list=None):
        n_plots = len(self.actual)
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 5 * n_plots))

        if n_plots == 1:
            axes = [axes]  # Ensure axes is always a list

        for i, ax in enumerate(axes):
            ax.plot(self.time[i], self.actual[i], label='Actual', color='blue')
            ax.plot(self.time[i], self.predicted[i], label='Predicted', color='orange')
            ax.fill_between(self.time[i], self.lower_ci[i], self.upper_ci[i], color='gray', alpha=0.2,
                            label='Confidence Interval')

            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend(loc='upper left')

            if title_list and i < len(title_list):
                ax.set_title(title_list[i])
            else:
                ax.set_title(f'Time Series Plot {i + 1}')

        plt.tight_layout()
        plt.savefig('figs/times series plot C.I..png')
        plt.show()