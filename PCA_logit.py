# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 20:11:38 2019

@author: hp
"""


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import zscore

data=pd.read_csv("Data.csv")
data.shape
data.dtypes

data.describe()

#count by class
data.groupby(["Creditability"]).count()

X_data=data.drop(labels="Creditability",axis=1)

X_data_z=X_data.apply(zscore)

X = np.array(X_data_z)

Y_data=data["Creditability"]
y= np.array(Y_data)

cov_matrix = np.cov(X.T)
print('Covariance Matrix \n%s', cov_matrix)

# Determine the eigen value and vector
# Use thr linear also function to do this
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)

print('Eigen Vectors \n%s',eig_vecs)
print('\n Eigen values \n%s', eig_vals)
# how many PCs should we have?? - eigen values more than 1 and 80-90% variance

# pair eigen values with eigen vectors
eigen_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
eigen_pairs

tot = sum(eig_vals)
# Need to sort it before you add it
var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)]
# Calculate cumulative variance
cum_var_exp = np.cumsum(var_exp)
print("Cumulative Variance Explained", cum_var_exp)
# from this we see just picking eigen value more than1 (kaisers criteria) is not sufficient
# but that is only 8 PCs, which explains only 59.2%. 
# to explain atleast 80% variance - we need to select 13 PCs
plt.figure(figsize=(6,2))
plt.bar(range(4), var_exp, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(4), cum_var_exp, where='mid',label='Cumulative explained variance')
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.legend(loc = "best")
plt.tight_layout()
plt.show()

#Extract the PC components
#Reduce dimensions from 21 to 13
#Compute PCA scores
# These scores are now input to logistic regression
pca = PCA(n_components = 13)
pca_scores = pca.fit_transform(X)
pca_scores[:1,]

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(pca_scores,y)
# if you run without PCA with original data, the instead of pca_scores in above command, it will be X)
#sklearn automaticallt uses 0.5 as threshold

# Predicting the Test set results
y_pred = classifier.predict(pca_scores)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred)


627+144
771/1000








