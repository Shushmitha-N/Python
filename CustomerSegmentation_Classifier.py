# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np

train =pd.read_csv('train.csv')
test =pd.read_csv('test.csv')
train.head()
test.head()

key = pd.DataFrame(columns=['ID','Segmentation'])
key
key.ID.append(train.loc[train.ID==467358]['Segmentation'])
key
a = train.ID.isin(test.ID)
final_test = test[test.ID.isin(train.ID)==False]
final_test
train.describe()
p = pd.merge( test[['ID']] ,  train[['ID','Segmentation']] , on='ID', how ='left' )
p
p_ids_null = p[p.Segmentation.isnull()]['ID']
real_y = pd.merge(p_ids_null,  test , on='ID', how ='left' )
df=pd.concat([train,real_y],ignore_index=True, sort=False)

# NULL VALUES

df[df.Ever_Married.isnull()].isnull().sum()
df[df.Ever_Married.isnull()].Spending_Score.value_counts()
# High, Avg spending - married, Graduated - married
df.loc[ (pd.isnull(df['Ever_Married'])) & (df['Spending_Score'] != 'Low'), 'Ever_Married'] = 'Yes'
df.loc[ (pd.isnull(df['Ever_Married'])) & (df['Graduated'] == 'Yes'), 'Ever_Married'] = 'Yes'
df.loc[ (pd.isnull(df['Ever_Married'])) & (df['Graduated'] == 'No'), 'Ever_Married'] = 'No'
df[df.Ever_Married.isnull()].isnull().sum()

df[df.Graduated.isnull()].isnull().sum()
# married - graduated
df.loc[ (pd.isnull(df['Graduated'])) & (df['Ever_Married'] == 'Yes'), 'Graduated'] = 'Yes'
df.loc[ (pd.isnull(df['Graduated'])) & (df['Ever_Married'] == 'No'), 'Graduated'] = 'No'
df[df.Graduated.isnull()].isnull().sum()

df[df.Profession.isnull()].isnull().sum()
df.Profession.fillna(method='ffill',inplace=True)

df.Var_1.fillna(method='ffill',inplace=True)

for i in ['Healthcare', 'Engineer', 'Lawyer', 'Entertainment', 'Artist','Executive', 'Doctor', 'Homemaker', 'Marketing']:
    print(i,'\n',df[df.Profession == i]['Work_Experience'].median(),'\n')
df.loc[ (pd.isnull(df['Work_Experience'])) & (df['Profession'] == 'Healthcare'), 'Work_Experience'] = 1
df.loc[ (pd.isnull(df['Work_Experience'])) & (df['Profession'] == 'Engineer'), 'Work_Experience'] = 1
df.loc[ (pd.isnull(df['Work_Experience'])) & (df['Profession'] == 'Lawyer'), 'Work_Experience'] = 1
df.loc[ (pd.isnull(df['Work_Experience'])) & (df['Profession'] == 'Entertainment'), 'Work_Experience'] = 1
df.loc[ (pd.isnull(df['Work_Experience'])) & (df['Profession'] == 'Artist'), 'Work_Experience'] = 1
df.loc[ (pd.isnull(df['Work_Experience'])) & (df['Profession'] == 'Executive'), 'Work_Experience'] = 1
df.loc[ (pd.isnull(df['Work_Experience'])) & (df['Profession'] == 'Doctor'), 'Work_Experience'] = 1
df.loc[ (pd.isnull(df['Work_Experience'])) & (df['Profession'] == 'Homemaker'), 'Work_Experience'] = 8
df.loc[ (pd.isnull(df['Work_Experience'])) & (df['Profession'] == 'Marketing'), 'Work_Experience'] = 1

#ever_married = yes , family size = 2
df.loc[ (pd.isnull(df['Family_Size'])) & (df['Ever_Married'] == 'Yes'), 'Family_Size'] = 2
df.loc[ (pd.isnull(df['Family_Size'])) & (df['Ever_Married'] == 'No'), 'Family_Size'] = 1

df.isnull().sum()

# Labelling - Categirical Variables
def gen(x):
    if x=='Male':
        x=1
    else:
        x=0
    return(x)
df.Gender = df.Gender.apply(gen)

def y_or_n(x):
    if x=='Yes':
        x=1
    else:
        x=0
    return(x)
df.Ever_Married = df.Ever_Married.apply(y_or_n)
df.Graduated = df.Graduated.apply(y_or_n)
df.head()
df=pd.get_dummies(df,columns=['Profession','Var_1','Spending_Score'])

# Train data
traindata = df.dropna() # to drop rows with na in segmentation
X = traindata.drop(['Segmentation','ID'],axis=1)
y = traindata['Segmentation']
def target(x):
    if x == 'A':
        x=1
    elif x == 'B':
        x=2
    elif x == 'C':
        x=3
    elif x == 'D':
        x=4
    return(x)
y = y.apply(target)

# Using SMOTE to balance
from imblearn.over_sampling import SMOTE
smote = SMOTE('auto')
X_sm, y_sm = smote.fit_sample(X,y)
print(X_sm.shape, y_sm.shape)

# Train Test split
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.3, random_state=42)

# XGB
from xgboost import XGBClassifier
modelxg = XGBClassifier(max_depth=100,learning_rate=0.1,n_estimators=150, random_state=42)
modelxg.fit(X_train, y_train)
y_pred_xg = modelxg.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred_xg)

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
grid = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }
kf = KFold(n_splits=2)
gs = GridSearchCV(estimator = XGBClassifier(n_estimators=500), param_grid = grid, scoring='accuracy',n_jobs=4, cv=kf)
gs.fit(X_train, y_train)
y_pred_xgcv = gs.predict(X_test)
gs.best_estimator_
xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1.0, gamma=2, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.300000012, max_delta_step=0, max_depth=5,
              min_child_weight = 5, monotone_constraints='()',
              n_estimators=500, n_jobs=0, num_parallel_tree=1,
              objective='multi:softprob', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=None, subsample=1.0,
              tree_method='exact', validate_parameters=1, verbosity=None)
xgb.fit(X_train, y_train)
y_pred_xgcv = xgb.predict(X_test)

accuracy_score(y_test, y_pred_xgcv)

X_train.columns
xgb.feature_importances_

# Randomforest
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(max_depth=150, min_samples_leaf= 5, min_samples_split= 3 , n_estimators=300, random_state = 42) 
forest.fit(X_train,y_train)
y_pred_rf = forest.predict(X_test)

accuracy_score(y_test, y_pred_rf)

# Deep Learning Classifier
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping


classifier = Sequential()

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 25))
# Second Layer
classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'uniform'))
# Output layer
classifier.add(Dense(units = 1, activation = 'softmax', kernel_initializer = 'uniform'))
# if it has to be a multi group classification use 'softmax' function

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Optimiser is for gradient descent - in order to make sure that it finds global minima and not local minima

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

y_pred_dl = classifier.predict(X_test)
accuracy_score(y_test,y_pred_dl)








