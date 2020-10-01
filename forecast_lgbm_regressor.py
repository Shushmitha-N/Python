# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import lightgbm as lgbm
from sklearn.preprocessing import LabelEncoder

#Correlation matrix, tree diagram, feature importance


test=pd.read_csv('test.csv')
train=pd.read_csv('train.csv')
center=pd.read_csv('fulfilment_center_info.csv')
meal=pd.read_csv('meal_info.csv')

train.shape,center.shape,meal.shape,test.shape
train.head()
center.head()
meal.head()
test.head()

train['week_num']=train.week%52
test['week_num']=test.week%52
train.tail()

train=pd.merge(train,center,how='left',on=['center_id'])
test=pd.merge(test,center,how='left',on=['center_id'])
train=pd.merge(train,meal,how='left',on=['meal_id'])
test=pd.merge(test,meal,how='left',on=['meal_id'])
train.shape,test.shape

# Features
train['discount']=train.base_price-train.checkout_price
test['discount']=test.base_price-test.checkout_price

def count_of_ids(train,test,col,name):
    temp=train.groupby(col)['id'].count().reset_index().rename(columns={'id':name})
    train=pd.merge(train,temp,how='left',on=col)
    test=pd.merge(test,temp,how='left',on=col)
    train[name]=train[name].astype(float)
    test[name]=test[name].astype(float)
    train[name].fillna(np.median(temp[name]),inplace=True)
    test[name].fillna(np.median(temp[name]),inplace=True)
    return train,test

train,test = count_of_ids(train,test,col=['meal_id','center_id'],name='meal&center_id_count')
train,test = count_of_ids(train,test,col=['cuisine','center_id'],name='cuisine&center_id_count')
train,test = count_of_ids(train,test,col=['meal_id'],name='meal_id_count')
train,test = count_of_ids(train,test,col=['category','center_id'],name='category&center_id_count')
train,test = count_of_ids(train,test,col=['center_id'],name='center_id_count')


def avg_price(train,test,col,price='base_price',name='name'):
    temp=train.groupby(col)[price].mean().reset_index().rename(columns={price:name})
    train=pd.merge(train,temp,how='left',on=col)
    test=pd.merge(test,temp,how='left',on=col)
    train[name].fillna(np.median(temp[name]),inplace=True)
    test[name].fillna(np.median(temp[name]),inplace=True)
    return train,test

train,test = avg_price(train,test,col=['meal_id','center_id'],price='base_price', name='avg_price_bp_meal&center')
train,test = avg_price(train,test,col=['meal_id','center_id'],price='checkout_price', name='avg_price_cp_meal&center')
train,test = avg_price(train,test,col=['center_id','cuisine'],price='base_price', name='avg_price_bp_center&cuisine')
train,test = avg_price(train,test,col=['center_id','cuisine'],price='checkout_price', name='avg_price_cp_center&cuisine')
train,test = avg_price(train,test,col=['category','region_code'],price='base_price', name='avg_price_bp_category&region')
train,test = avg_price(train,test,col=['category','region_code'],price='checkout_price', name='avg_price_cp_category&region')

def order_count_mean(train,test,col,name):
    temp=train.groupby(col)['num_orders'].mean().reset_index().rename(columns={'num_orders':name})
    train=pd.merge(train,temp,how='left',on=col)
    test=pd.merge(test,temp,how='left',on=col)
    train[name].fillna(np.median(temp[name]),inplace=True)
    test[name].fillna(np.median(temp[name]),inplace=True)
    return train,test

train,test = order_count_mean(train,test,col=['meal_id','center_id'],name='meal&center_mean_orders')
train,test = order_count_mean(train,test,col=['center_id','category','cuisine'],name='center&category&cuisine_mean_orders')
train,test = order_count_mean(train,test,col=['cuisine','category'],name='cuisine&category_mean_orders')
train,test = order_count_mean(train,test,col=['center_id','cuisine'],name='center&cuisine_mean_orders')
train,test = order_count_mean(train,test,col=['center_id','category'],name='center&category_mean_orders')
train,test = order_count_mean(train,test,col=['city_code','cuisine'],name='city&cuisine_mean_orders')
train,test = order_count_mean(train,test,col=['city_code','region_code'],name='city&region_mean_orders')
train,test = order_count_mean(train,test,col=['center_id'],name='center_mean_orders')
train,test = order_count_mean(train,test,col=['meal_id'],name='meal_mean_orders')
train,test = order_count_mean(train,test,col=['category','region_code'],name='category&region_mean_orders')
train,test = order_count_mean(train,test,col=['cuisine','region_code'],name='cuisine&region_mean_orders')

# Encoding Categorical Variables
for i in train.columns :
    if train[i].dtypes=='object':
        print(i)
        le=LabelEncoder()
        train[i]=le.fit_transform(train[i])
        test[i]=le.transform(test[i])
        
train.describe()
train.dtypes
test.dtypes

# Final train
X=train.drop(['id','week','num_orders','center_id','meal_id'],axis=1)
Y=train.num_orders
X.shape,Y.shape
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Evaluation metric
from sklearn.metrics import mean_squared_log_error
from math import sqrt
def rmsle(y_true, y_pred):
    return 100*sqrt(mean_squared_log_error(y_true, y_pred))

# Plot correlation
import seaborn as sns    
plt.subplots(figsize=(14,9))
data = X_train.corr()
sns.heatmap(data, cmap ='Blues')

# Modelling - regression

# Lightgbm
def train_lgb(max_depth=5,seed=4,num_round=2500):
    lgbm_train = lgbm.Dataset(X_train,y_train)
    params = {
        'objective' :'regression',
        'max_depth':max_depth,
        'learning_rate' : 0.01,
        'num_leaves' :(2*max_depth)-1 ,
        'feature_fraction': 0.8,
        "min_data_in_leaf" : 100,
        'bagging_fraction': 0.7, 
        'boosting_type' : 'gbdt',
        'metric': 'rmse',
        'seed':seed
    }
    lgb= lgbm.train(params, lgbm_train, num_round)
    return lgb

lgb1=train_lgb(7,5,2750)

y_pred=lgb1.predict(X_test)
y_pred = abs(y_pred)
rmsle(y_test,y_pred)

# Feature importance
# _,ax = plt.subplots(1,1,figsize=(12,6))
lgbm.plot_importance(lgb1)






















