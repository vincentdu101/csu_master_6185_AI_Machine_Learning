# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import graphviz
from sklearn import tree
from sklearn.datasets import load_iris

# import dataset
dataset = pd.read_csv("S.csv")

# dataset.iloc[row_selection, column_selection]
X = dataset.iloc[:, 1:5].values
y = dataset.iloc[:, 5].values

# encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
outlookEncoder = LabelEncoder()
temperatureEncoder = LabelEncoder()
humidityEncoder = LabelEncoder()
windEncoder = LabelEncoder()
yEncoder = LabelEncoder()

# encode independent variables 
outlookEncoder.fit(["Sunny", "Overcast", "Rain"])
X[:, 0] = outlookEncoder.transform(X[:, 0])

temperatureEncoder.fit(["Hot", "Mild", "Cool"])
X[:, 1] = temperatureEncoder.transform(X[:, 1])

humidityEncoder.fit(["High", "Normal"])
X[:, 2] = humidityEncoder.transform(X[:, 2])

windEncoder.fit(["Weak", "Strong"])
X[:, 3] = windEncoder.transform(X[:, 3])

# encode dependent variable
yEncoder.fit(["No", "Yes"])
y[:] = yEncoder.transform(y[:])

# fit training and test data together
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y.astype(int))

test = clf.predict([[0, 1, 1, 0]])
dot_data = tree.export_graphviz(clf, out_file = None,
                                feature_names = ["outlook", "temperature", "humidity", "wind"],
                                class_names = "playball",
                                filled=True, rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("question_1")
graph


