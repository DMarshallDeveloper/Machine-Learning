import random
import numpy as np
from Part3.simple_linear_regression import load_data, learn_and_analyse, data_preprocess
from Part3.utilities.visualization import visualize

seed = 309
# Freeze the random seed
random.seed(seed)
np.random.seed(seed)
train_test_split_test_size = 0.3

# Training settings
alpha = 0.1  # step size
max_iters = 50  # max iterations
fileName = "data/Part2.csv"

# Step 1: Load Data
data = load_data(fileName)

# Step 2: Preprocess the data
train_data, train_labels, test_data, test_labels, train_data_full, test_data_full = data_preprocess(data,
                                                                                                    train_test_split_test_size)

# Step 3: Learn and process the results
# This goes through the four algorithms, prints out their learn execution time,
# R2, MSE, RMSE, and MAE and the thetas and losses for each iteration of each algorithm
# It then calls visualize which generates 4 graphs and saves them to the figures folder


metric_type = "MSE"  # MSE, RMSE, MAE, R2
optimizer_type = "BGD"  # PSO, BGD
thetas, losses = learn_and_analyse(train_data, train_labels, test_data, test_labels, optimizer_type, metric_type,
                                   max_iters, alpha)
visualize(train_data_full, train_labels, train_data, thetas, losses, max_iters, test_data_full, test_data,
          optimizer_type, metric_type, fileName)

metric_type = "MSE"  # MSE, RMSE, MAE, R2
optimizer_type = "MiniBGD"  # PSO, BGD
thetas, losses = learn_and_analyse(train_data, train_labels, test_data, test_labels, optimizer_type, metric_type,
                                   max_iters, alpha)
visualize(train_data_full, train_labels, train_data, thetas, losses, max_iters, test_data_full, test_data,
          optimizer_type, metric_type, fileName)

metric_type = "MSE"  # MSE, RMSE, MAE, R2
optimizer_type = "PSO"  # PSO, BGD
thetas, losses = learn_and_analyse(train_data, train_labels, test_data, test_labels, optimizer_type, metric_type,
                                   max_iters, alpha)
visualize(train_data_full, train_labels, train_data, thetas, losses, max_iters, test_data_full, test_data,
          optimizer_type, metric_type, fileName)

metric_type = "MAE"  # MSE, RMSE, MAE, R2
optimizer_type = "PSO"  # PSO, BGD
thetas, losses = learn_and_analyse(train_data, train_labels, test_data, test_labels, optimizer_type, metric_type,
                                   max_iters, alpha)
visualize(train_data_full, train_labels, train_data, thetas, losses, max_iters, test_data_full, test_data,
          optimizer_type, metric_type, fileName)

fileName = "data/Part2Outliers.csv"

# Step 1: Load Data
data = load_data(fileName)

# Step 2: Preprocess the data
train_data, train_labels, test_data, test_labels, train_data_full, test_data_full = data_preprocess(data,
                                                                                                    train_test_split_test_size)

metric_type = "MSE"  # MSE, RMSE, MAE, R2
optimizer_type = "PSO"  # PSO, BGD
thetas, losses = learn_and_analyse(train_data, train_labels, test_data, test_labels, optimizer_type, metric_type,
                                   max_iters, alpha)
visualize(train_data_full, train_labels, train_data, thetas, losses, max_iters, test_data_full, test_data,
          optimizer_type, metric_type, fileName)

metric_type = "MAE"  # MSE, RMSE, MAE, R2
optimizer_type = "PSO"  # PSO, BGD
thetas, losses = learn_and_analyse(train_data, train_labels, test_data, test_labels, optimizer_type, metric_type,
                                   max_iters, alpha)
visualize(train_data_full, train_labels, train_data, thetas, losses, max_iters, test_data_full, test_data,
          optimizer_type, metric_type, fileName)
