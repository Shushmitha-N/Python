# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 18:21:53 2019

@author: Susmita
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset
dataset = pd.read_csv("Churn_Modelling.csv")

# Splitting into dependent and independent variable
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values
dataset

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# First variable is country
labelencoder_X_1 = LabelEncoder() 
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) # transforming country column
# Second variable is gender
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
# Fixing the dummy variable trap
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] # to drop the first column - unnecessary dummy variable

# Split into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling - since credit score is in one dimension, others are in different dimensions
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) # fit_transform and also trnsform will work


# Importing the Keras libraries and packages
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
# units - sets up the number of units in the hidden layer - can be set by taking avg of input and output - (11+1)/2 = 6
# Kernal initialiser - Uniform distribution set for initial weights of the synapses
# Activation - activation function in hidden layer
# input_dim - number of variables that are inputs
# Dense function implements the operation - output = activation(dot(input,kernal)+bias) which is activation function of sumproduct plus bias

# Second Layer
classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'uniform'))

# Output layer
classifier.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'uniform'))
# Here ouputactivation function is sigmoid and not relu because we need the output to be a binary classification
# if it has to be a multi group classification use 'softmax' function

# to compare the output of the output layer with the actual output - we need a loss function
# Binary cross entropy loss function is used when it is a single classification problem.
# there are 45 different loss functions
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Optimiser is for gradient descent - in order to make sure that it finds global minima and not local minima

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
# Why getting different accuracy? initialising weights will be different for everyone so convergence will be different
# "Black box conundrum"

# Making the predictions and evaluating the model
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Running a prediction on new data point
# This should be done by creating a numpy array - difference between dataframe and numpy array?
# France, credit score 600, male, age 40, tenure 3 yrs, balance 60000, no of products 2, has credit card, is active member, salary 50000)
new_customer = classifier.predict(sc.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))
new_prediction = (new_customer>0.5)

# Evaluation and cross validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def build_classifier():
  classifier = Sequential()
  classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
  classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'uniform'))
  classifier.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'uniform'))
  classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])  
  return classifier

# Once the classifier has been created, enter batch size and epochs
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)
mean = accuracies.mean()
variance = accuracies.std()

# Improve/Tuning the ANN
# Grid search - a methodology to tune your network
# It helps to tune all parameters that will methodically build and evaluate a model for each combination 
# will come up with a combination for best accuracy

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['acc'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)

# Create parameter as a dictionary. batch_size, epochs and optimizer are keys. 
# The grid search will try to optimise within the combination of all these parameters
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}

# Grid search object and then data is fitted on it
# GridSearchCV is a combination of grid search and cv
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)

# whenever you are looking for parameters you need to add "_" in the end or it wont locate the parameter
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_



