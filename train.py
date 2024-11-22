#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import pickle

print('Preparing data...')
df = pd.read_csv('water_potability.csv')

df.columns = df.columns.str.lower()
df = df.dropna(subset=['ph'])
df = df.dropna(subset=['trihalomethanes'])
df['sulfate'] = df.groupby('potability')['sulfate'].transform(lambda x: x.fillna(x.median()))


chosen_features = ['ph', 'hardness', 'solids', 'sulfate', 'conductivity',
       'turbidity', 'potability'
       ]
df = df[chosen_features]



X = df.drop(columns=['potability'])
y = df['potability']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


print('Training model...')
features = ['ph', 'hardness', 'solids', 'sulfate', 'conductivity',
       'turbidity']
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)



xgb_params = {
    'eta': 0.3, 
    'max_depth': 6,
    'min_child_weight': 1,
    
    'objective': 'binary:logistic',
    'nthread': 8,
    
    'seed': 2,
    'verbosity': 1,
}

xgbmodel = xgb.train(xgb_params, dtrain, num_boost_round=10)
y_pred = xgbmodel.predict(dtest)

print('auc score= ', roc_auc_score(y_test, y_pred))


# Save the model
output_file = 'water_model.bin'
with open(output_file, 'wb') as f_out :
    pickle.dump(xgbmodel, f_out)

print('Done!')


