import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class RegressionNN:
    def __init__(self, input_dim, hidden, output_dim, learning_rate=0.01):
        self.model = self.build_model(input_dim, hidden, output_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def build_model(self, input_dim, hidden, output_dim):
        class SimpleNN(nn.Module):
            def __init__(self):
                super(SimpleNN, self).__init__()
                self.fc1 = nn.Linear(input_dim, hidden[0])
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(hidden[0], hidden[1])
                self.relu = nn.ReLU()
                self.fc3 = nn.Linear(hidden[1], output_dim)

            def forward(self, x):
                out = self.fc1(x)
                out = self.relu(out)
                out = self.fc2(out)
                out = self.relu(out)
                out = self.fc3(out)
                return out

        return SimpleNN()

    def train(self, x_train, y_train, num_epochs=1000, print_interval=1000):
        for epoch in range(num_epochs):
            # Forward pass
            outputs = self.model(x_train)
            loss = self.criterion(outputs, y_train)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % print_interval == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    def evaluate(self, x):
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            predicted = self.model(x).numpy()
        return predicted

    def calculate_metrics(self, y_train, y_pred):
        # Convert tensors to numpy arrays
        y_true = y_train.numpy()
        # Calculate RMSE
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        # Calculate R^2
        ss_residual = np.sum((y_true - y_pred) ** 2)
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        return rmse, r2

    def plot_results(self, x_train, y_train, predicted):
        plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original data')
        plt.plot(x_train.numpy(), predicted, 'bo', label='Fitted line')
        plt.legend()
        plt.show()


# %%
# Run the model
x_train = pd.read_csv("Dataset for metamodeling/X_train_dev.csv")
y_train = pd.read_csv("Dataset for metamodeling/y_train_dev.csv")

# Convert numpy arrays to torch tensors
x_train = torch.tensor(x_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)

# Hyperparameters
input_dim = x_train.shape[1]
hidden = [512, 32]  # [512, 512] RMSE 0.08, R2 0.88
output_dim = y_train.shape[1]
learning_rate = 0.02
num_epochs = 1000

# Initialize and train the model
regression_nn = RegressionNN(input_dim, hidden, output_dim, learning_rate)
regression_nn.train(x_train, y_train, num_epochs)

# Evaluate the model
y_pred = regression_nn.evaluate(x_train)
print(y_pred)
# Results
# Calculate RMSE and R^2
rmse, r2 = regression_nn.calculate_metrics(y_train, y_pred)
print(f'RMSE: {rmse:.4f}, R^2: {r2:.4f}')
