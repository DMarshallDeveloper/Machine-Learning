import datetime
import random

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, \
    AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

trainFileName = "data/adult.data"
testFileName = "data/adult.test"
target = "income"


# Step 1: Load Data
train_data = pd.read_table(trainFileName, sep="\t", header=None, names=["age", "workclass", "fnlwgt",
                                                                        "education", "education-num", "marital-status",
                                                                        "occupation", "relationship", "race", "sex",
                                                                        "capital-gain", "capital-loss",
                                                                        "hours-per-week", "native-country", "income"])

test_data = pd.read_table(testFileName, sep=", ", skiprows=1, header=None, names=["age", "workclass", "fnlwgt",
                                                                                  "education", "education-num",
                                                                                  "marital-status",
                                                                                  "occupation", "relationship", "race",
                                                                                  "sex",
                                                                                  "capital-gain", "capital-loss",
                                                                                  "hours-per-week", "native-country",
                                                                                  "income"], engine='python')

modeWorkclass = train_data['workclass'].value_counts().idxmax()
modeOccupation = train_data['occupation'].value_counts().idxmax()
modeNativeCountry = train_data['native-country'].value_counts().idxmax()

train_data = train_data.replace(to_replace={"workclass": {"?": modeWorkclass},
                                            "occupation": {"?": modeOccupation},
                                            "native-country": {"?": modeNativeCountry}
                                            }, value=None)
test_data = test_data.replace(to_replace={"workclass": {"?": modeWorkclass},
                                          "occupation": {"?": modeOccupation},
                                          "native-country": {"?": modeNativeCountry}
                                          }, value=None)

# Check the datatype of each variable
print("\nData Types \n", train_data.dtypes)

# Check the shape of the data
print("\nTrain Shape: ", train_data.shape)
print("Test Shape: ", test_data.shape)

# trainData.to_csv("../data/Part 2 - classification/export.csv")


# Pre-process data (both train and test)
train_data_full = train_data.copy()
train_data = train_data.drop([target], axis=1)
train_labels = train_data_full[target]

test_data_full = test_data.copy()
test_data = test_data.drop([target], axis=1)
test_labels = test_data_full[target]
test_labels = test_labels.replace(to_replace={"<=50K.": " <=50K", ">50K.": " >50K"}, value=None)

# Join them together to create the binary columns to ensure that there
# is a binary column, for every variable in the training set, in the test set
# Then separate them again
full_data = pd.concat((train_data, test_data))
full_data = pd.get_dummies(full_data, drop_first=True)
train_data = full_data.iloc[:32561, :]
test_data = full_data.iloc[32561:, :]


# Fit each algorithm to the training data, then calculate each metric and print the results
def fitAndTest(model):
    start_time = datetime.datetime.now()  # Track learning starting time
    model.fit(train_data, train_labels)
    predictions = model.predict(test_data)
    predictions = pd.Series(predictions)

    ROC_test_labels = test_labels.replace(to_replace={" <=50K": "0", " >50K": "1"}, value=None)
    ROC_predictions = predictions.replace(to_replace={" <=50K": "0", " >50K": "1"}, value=None)
    ROC_test_labels = ROC_test_labels.astype(float)
    ROC_predictions = ROC_predictions.astype(float)

    print("Classification Accuracy:", metrics.accuracy_score(test_labels, predictions))
    print("Precision:", metrics.precision_score(test_labels, predictions, pos_label=' >50K'))
    print("Recall:", metrics.recall_score(test_labels, predictions, pos_label=' >50K'))
    print("F1-Score:", metrics.f1_score(test_labels, predictions, pos_label=' >50K'))
    print("ROC AUC:", metrics.roc_auc_score(ROC_test_labels, ROC_predictions))
    end_time = datetime.datetime.now()  # Track learning ending time
    execution_time = (end_time - start_time).total_seconds()  # Track execution time
    print("Learn: execution time={t:.3f} seconds".format(t=execution_time))


# Step 3: Run each algorithm and print the results
model = KNeighborsClassifier()
print("\n KNeighborsClassifier: \n")
fitAndTest(model)
model = GaussianNB()
print("\n GaussianNB: \n")
fitAndTest(model)
model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
print("\n SVC: \n")
fitAndTest(model)
model = DecisionTreeClassifier(random_state=0)
print("\n DecisionTreeClassifier: \n")
fitAndTest(model)
model = RandomForestClassifier(max_depth=20, random_state=0)
print("\n RandomForestClassifier: \n")
fitAndTest(model)
model = AdaBoostClassifier(n_estimators=100, random_state=0)
print("\n AdaBoostClassifier: \n")
fitAndTest(model)
model = GradientBoostingClassifier(random_state=0)
print("\n GradientBoostingClassifier: \n")
fitAndTest(model)
model = LinearDiscriminantAnalysis()
print("\n LinearDiscriminantAnalysis: \n")
fitAndTest(model)
model = MLPClassifier(random_state=1, max_iter=1000)
print("\n MLPClassifier: \n")
fitAndTest(model)
model = LogisticRegression(random_state=0)
print("\n LogisticRegression: \n")
fitAndTest(model)
