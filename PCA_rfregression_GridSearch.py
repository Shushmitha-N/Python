# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 22:39:08 2020

@author: shushmitha natarajan
"""



# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Data
dataset = pd.read_csv('Train.csv')
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values


# Building the model

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
y_train1 = sc.fit_transform(y_train.reshape(-1,1))
y_test1 = sc.fit_transform(y_test.reshape(-1,1))

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 3)
X_train1 = pca.fit_transform(X_train)
X_test1 = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_


# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
#regressor = RandomForestRegressor(n_estimators = 50, random_state = 0)
regressor = RandomForestRegressor(n_estimators = 30, random_state = 0, max_features = "auto", max_depth = 5, max_leaf_nodes = 10, max_samples = 15, bootstrap = "False")
regressor.fit(X_train1, y_train)

# Predicting 
y_pred = regressor.predict(X_test1)
import sklearn.metrics
sorted(sklearn.metrics.SCORERS.keys()) 

#Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators': [5,10,12,15,18,20,30,40,45,50], 'max_features' : ["auto", 'sqrt'], 'max_depth' : [2,5,10], 'max_leaf_nodes' : [2,5,8,10], 'max_samples' : [2,5,8,10,15], 'bootstrap' : ['False',"True"]}]
grid_search = GridSearchCV(estimator = regressor, param_grid = parameters, scoring = 'neg_root_mean_squared_error', cv = 10, n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

import math
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)

rmse = math.sqrt(mse)

print(rmse)

# On test data 
Test_data = dataset = pd.read_csv('Test.csv')
Test_pca = pca.fit_transform(Test_data)
y_pred_test = regressor.predict(Test_pca)
