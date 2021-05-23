# -*- coding: utf-8 -*-

"""
Visualization functions
"""

import matplotlib.pyplot as plt
import numpy as np

# Visualize the training course
from Part3.utilities.losses import compute_loss


def compute_z_loss(y, x, thetas):
    """
    Compute z-axis values
    :param y:            train labels
    :param x:            train data
    :param thetas:       model parameters
    :return: z_losses    value (loss) for z-axis
    """
    thetas = np.array(thetas)
    w = thetas[:, 0].reshape(thetas[:, 0].shape[0], )
    b = thetas[:, 1].reshape(thetas[:, 1].shape[0], )
    z_losses = np.zeros((len(w), len(b)))
    for ind_row, row in enumerate(w):
        for ind_col, col in enumerate(b):
            theta = np.array([row, col])
            z_losses[ind_row, ind_col] = compute_loss(y, x, theta, "MSE")
    return z_losses


def predict(x, thetas):
    """
    Predict function
    :param x:               test data
    :param thetas:          trained model parameters
    :return:                predicted labels
    """
    return x.dot(thetas)


def visualize(train_data_full, train_labels, train_data, thetas, losses, niter, test_data_full, test_data,
              optimizer_type, metric_type, fileName):
    """
    Visualize Function for Training Results
    :param train_data:
    :param train_labels:
    :param train_data_full:   the train data set (full) with labels and data
    :param thetas:            model parameters
    :param losses:            all tracked losses
    :param niter:             completed training iterations
    :return: fig1              the figure for line fitting on training data
             fig2              learning curve in terms of error
             fig3              gradient variation visualization
    """

    # i is used to determine the figure number, i.e. Fig 1.1 or fig 2.3,
    # to make the report analysis clearer
    i = "0"
    if optimizer_type == "BGD" and metric_type == "MSE":
        i = "1"
    elif optimizer_type == "MiniBGD" and metric_type == "MSE":
        i = "2"
    elif optimizer_type == "PSO" and metric_type == "MSE" and fileName == "data/Part2.csv":
        i = "3"
    elif optimizer_type == "PSO" and metric_type == "MAE" and fileName == "data/Part2.csv":
        i = "4"
    elif optimizer_type == "PSO" and metric_type == "MSE" and fileName == "data/Part2Outliers.csv":
        i = "5"
    elif optimizer_type == "PSO" and metric_type == "MAE" and fileName == "data/Part2Outliers.csv":
        i = "6"

    fig = plt.figure(figsize=(14, 9))

    ax_test = fig.add_subplot(2, 2, 1)
    ax_test.scatter(test_data_full["Weight"], test_data_full["Height"], color='blue')
    ax_test.plot(test_data_full["Weight"], predict(test_data, thetas[-1]), color='red', linewidth=2)
    ax_test.set_xlabel("Height")
    ax_test.set_ylabel("Weight")
    ax_test.set_title("Fig " + i + ".1: " + "Test Height vs Weight")

    ax_1 = fig.add_subplot(2, 2, 2)
    ax_1.scatter(train_data_full["Weight"], train_data_full["Height"], color='blue')

    # De-standarize
    train_mean = train_data_full["Weight"].mean()
    train_std = train_data_full["Weight"].std()
    train_data_for_plot = train_mean + train_data["Weight"] * train_std

    ax_1.plot(train_data_for_plot, predict(train_data, thetas[niter - 1]), color='red', linewidth=2)
    ax_1.set_xlabel("Height")
    ax_1.set_ylabel("Weight")
    ax_1.set_title("Fig " + i + ".2: " + "Training Height vs Weight")

    ax_2 = fig.add_subplot(2, 2, 3)
    ax_2.plot(range(len(losses)), losses, color='blue', linewidth=2)
    ax_2.set_xlabel("Iteration")
    ax_2.set_ylabel("MSE")
    ax_2.set_title("Fig " + i + ".3: " + "Learning curve in terms of error")

    ax_3 = fig.add_subplot(2, 2, 4)
    np_gradient_ws = np.array(thetas)

    w = np.linspace(min(np_gradient_ws[:, 0]), max(np_gradient_ws[:, 0]), len(np_gradient_ws[:, 0]))
    b = np.linspace(min(np_gradient_ws[:, 1]), max(np_gradient_ws[:, 1]), len(np_gradient_ws[:, 1]))
    x, y = np.meshgrid(w, b)
    z = compute_z_loss(train_labels, train_data, np.stack((w, b)).T)
    cp = ax_3.contourf(x, y, z, cmap=plt.cm.jet)
    fig.colorbar(cp, ax=ax_3)
    ax_3.plot(3.54794951, 66.63949115837143, color='red', marker='*', markersize=20)
    if niter > 0:
        thetas_to_plot = np_gradient_ws[:niter]
    ax_3.plot(thetas_to_plot[:, 0], thetas_to_plot[:, 1], marker='o', color='w', markersize=10)
    ax_3.set_xlabel(r'$w$')
    ax_3.set_ylabel(r'$b$')
    ax_3.set_title("Fig " + i + ".4: " + "Gradient Variation Visualization")

    fig.suptitle("Fig " + i + ": Graphs for " + optimizer_type + "_" + metric_type)
    plt.savefig("figures/" + "Fig " + i + "- Graphs for " + optimizer_type + "_" + metric_type + ".png")
