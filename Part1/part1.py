import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor

fileName = "data/diamonds.csv"
train_test_split_test_size = 0.3
target = "price"

data = pd.read_csv(fileName)


# Note: the following function was inspired by https://financetrain.com/multivariate-linear-regression-in-python-with-scikit-learn-library/
def correlation(df, variables1, n_rows, n_cols):
    fig = plt.figure(figsize=(8, 6))
    # fig = plt.figure(figsize=(14,9))
    for i, var in enumerate(variables1):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        asset = df.loc[:, var]
        ax.scatter(df["price"], asset)
        ax.set_xlabel("price")
        ax.set_ylabel("{}".format(var))
        ax.set_title(var + " vs price")
    fig.tight_layout()
    plt.show()


# Uncomment the line below to get the scatterplots on page 1 of the report to show up
# correlation(diamonds, diamonds.columns[0:10], 5, 3)

# Check for null / empty values
print("Are there any null values? \n", data.isnull().any())

# Check the datatype of each variable
print("\nData Types \n", data.dtypes)

# Check the shape of the data
print("\nShape: ", data.shape)

data = pd.get_dummies(data, drop_first=True)
print("\nHead\n", data.head())

# Correlation matrix for price
print("\n======")
print("Correlation to price")
print(data.corr('pearson')["price"])


# Step 2: Preprocess the data
# This method is taken and modified from simple_linear_regression.py
def data_preprocess(preprocess_data, preprocess_target, preprocess_train_test_split_test_size):
    """
    Data preprocess:
        1. Split the entire dataset into train and test
        2. Split outputs and inputs
        3. Standardize train and test
        4. Add intercept dummy for computation convenience
    :param preprocess_target:
    :param preprocess_train_test_split_test_size:
    :param preprocess_data: the given dataset (format: panda DataFrame)
    :return: preprocess_train_data       train data contains only inputs
             preprocess_train_labels     train data contains only labels
             preprocess_test_data        test data contains only inputs
             preprocess_test_labels      test data contains only labels
             preprocess_train_data_full       train data (full) contains both inputs and labels
             preprocess_test_data_full       test data (full) contains both inputs and labels
    """
    # Split the data into train and test
    preprocess_train_data, preprocess_test_data = train_test_split(preprocess_data, test_size=preprocess_train_test_split_test_size)

    # Pre-process data (both train and test)
    preprocess_train_data_full = preprocess_train_data.copy()
    preprocess_train_data = preprocess_train_data.drop([preprocess_target], axis=1)
    preprocess_train_labels = preprocess_train_data_full[preprocess_target]

    preprocess_test_data_full = preprocess_test_data.copy()
    preprocess_test_data = preprocess_test_data.drop([preprocess_target], axis=1)
    preprocess_test_labels = preprocess_test_data_full[preprocess_target]

    # Standardize the inputs
    preprocess_train_mean = preprocess_train_data.mean()
    preprocess_train_std = preprocess_train_data.std()

    preprocess_train_data = (preprocess_train_data - preprocess_train_mean) / preprocess_train_std
    preprocess_test_data = (preprocess_test_data - preprocess_train_mean) / preprocess_train_std

    # Tricks: add dummy intercept to both train and test
    preprocess_train_data['intercept_dummy'] = pd.Series(1.0, index=preprocess_train_data.index)
    preprocess_test_data['intercept_dummy'] = pd.Series(1.0, index=preprocess_test_data.index)
    return preprocess_train_data, preprocess_train_labels, preprocess_test_data, preprocess_test_labels, preprocess_train_data_full, preprocess_test_data_full


#run the above method
train_data, train_labels, test_data, test_labels, train_data_full, test_data_full = data_preprocess(
    data, target, train_test_split_test_size)


# Fit each algorithm to the training data, then calculate each metric and print the results
def fit_and_test(fit_and_test_model):
    start_time = datetime.datetime.now()  # Track learning starting time
    fit_and_test_model.fit(train_data, train_labels)
    predictions = fit_and_test_model.predict(test_data)

    print("R2:", metrics.r2_score(test_labels, predictions))  # R2 should be maximize
    print("MSE:", metrics.mean_squared_error(test_labels, predictions))
    print("RMSE:", np.sqrt(metrics.mean_squared_error(test_labels, predictions)))
    print("MAE:", metrics.mean_absolute_error(test_labels, predictions))
    end_time = datetime.datetime.now()  # Track learning ending time
    execution_time = (end_time - start_time).total_seconds()  # Track execution time
    print("Learn: execution time={t:.3f} seconds".format(t=execution_time))


# Step 3: Run each algorithm and print the results
model = LinearRegression()
print("\n LinearRegression: \n")
fit_and_test(model)
model = KNeighborsRegressor()
print("\n KNeighborsRegressor: \n")
fit_and_test(model)
model = Ridge(alpha=1.0)
print("\n Ridge: \n")
fit_and_test(model)
model = DecisionTreeRegressor(random_state=0)
print("\n DecisionTreeRegressor: \n")
fit_and_test(model)
model = RandomForestRegressor(max_depth=2, random_state=0)
print("\n RandomForestRegressor: \n")
fit_and_test(model)
model = GradientBoostingRegressor(random_state=0)
print("\n GradientBoostingRegressor: \n")
fit_and_test(model)
model = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3))
print("\n SGDRegressor: \n")
fit_and_test(model)
model = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
print("\n SVR: \n")
fit_and_test(model)
model = make_pipeline(StandardScaler(), LinearSVR(random_state=0, tol=1e-5))
print("\n LinearSVR: \n")
fit_and_test(model)
model = MLPRegressor(random_state=1, max_iter=5000)
print("\n MLPRegressor: \n")
fit_and_test(model)
