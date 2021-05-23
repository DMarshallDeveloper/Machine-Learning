# -*- coding: utf-8 -*-

"""
This is an example to perform simple linear regression algorithm on the dataset (weight and height),
where x = weight and y = height.
"""
import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from Part3.utilities.losses import compute_loss
from Part3.utilities.optimizers import gradient_descent, mini_batch_gradient_descent, pso


def load_data(fileName):
    """
    Load Data from CSV
    :return: df    a panda data frame
    """
    df = pd.read_csv(fileName)
    return df


def data_preprocess(data, train_test_split_test_size):
    """
    Data preprocess:
        1. Split the entire dataset into train and test
        2. Split outputs and inputs
        3. Standardize train and test
        4. Add intercept dummy for computation convenience
    :param train_test_split_test_size:
    :param data: the given dataset (format: panda DataFrame)
    :return: train_data       train data contains only inputs
             train_labels     train data contains only labels
             test_data        test data contains only inputs
             test_labels      test data contains only labels
             train_data_full       train data (full) contains both inputs and labels
             test_data_full       test data (full) contains both inputs and labels
    """
    # Split the data into train and test
    train_data, test_data = train_test_split(data, test_size=train_test_split_test_size)

    # Pre-process data (both train and test)
    train_data_full = train_data.copy()
    train_data = train_data.drop(["Height"], axis=1)
    train_labels = train_data_full["Height"]

    test_data_full = test_data.copy()
    test_data = test_data.drop(["Height"], axis=1)
    test_labels = test_data_full["Height"]

    # Standardize the inputs
    train_mean = train_data.mean()
    train_std = train_data.std()
    train_data = (train_data - train_mean) / train_std
    test_data = (test_data - train_mean) / train_std

    # Tricks: add dummy intercept to both train and test
    train_data['intercept_dummy'] = pd.Series(1.0, index=train_data.index)
    test_data['intercept_dummy'] = pd.Series(1.0, index=test_data.index)
    return train_data, train_labels, test_data, test_labels, train_data_full, test_data_full


def learn(y, x, theta, max_iters, alpha, optimizer_type, metric_type):
    """
    Learn to estimate the regression parameters (i.e., w and b)
    :param y:                   train labels
    :param x:                   train data
    :param theta:               model parameter
    :param max_iters:           max training iterations
    :param alpha:               step size
    :param optimizer_type:      optimizer type (default: Batch Gradient Descient): GD, SGD, MiniBGD or PSO
    :param metric_type:         metric type (MSE, RMSE, R2, MAE). NOTE: MAE can't be optimized by GD methods.
    :return: thetas              all updated model parameters tracked during the learning course
             losses             all losses tracked during the learning course
    """
    thetas = None
    losses = None
    if optimizer_type == "BGD":
        thetas, losses = gradient_descent(y, x, theta, max_iters, alpha, metric_type)
    elif optimizer_type == "MiniBGD":
        thetas, losses = mini_batch_gradient_descent(y, x, theta, max_iters, alpha, metric_type, mini_batch_size=10)
    elif optimizer_type == "PSO":
        thetas, losses = pso(y, x, theta, max_iters, 100, metric_type)
    else:
        raise ValueError(
            "[ERROR] The optimizer '{ot}' is not defined, please double check and re-run your program.".format(
                ot=optimizer_type))
    return thetas, losses


def learn_and_analyse(train_data, train_labels, test_data, test_labels, optimizer_type, metric_type, max_iters, alpha):
    # Step 3: Learning Start
    theta = np.array([0.0, 0.0])  # Initialize model parameter

    start_time = datetime.datetime.now()  # Track learning starting time
    thetas, losses = learn(train_labels.values, train_data.values, theta, max_iters, alpha, optimizer_type, metric_type)

    end_time = datetime.datetime.now()  # Track learning ending time
    execution_time = (end_time - start_time).total_seconds()  # Track execution time

    # Step 4: Results presentation
    print("Learn: execution time={t:.3f} seconds".format(t=execution_time))

    print("R2:", -compute_loss(test_labels.values, test_data.values, thetas[-1], "R2"))  # R2 should be maximize
    print("MSE:", compute_loss(test_labels.values, test_data.values, thetas[-1], "MSE"))
    print("RMSE:", compute_loss(test_labels.values, test_data.values, thetas[-1], "RMSE"))
    print("MAE:", compute_loss(test_labels.values, test_data.values, thetas[-1], "MAE"))

    return thetas, losses
