#!/usr/bin/env python
# coding: utf-8

import pickle
import xgboost as xgb
import pandas as pd

input_file = "water_model.bin"

with open(input_file, 'rb') as f_in:
    model = pickle.load(f_in)


features = ['ph'	,'hardness', 'solids',  'sulfate' ,	'conductivity',  	'turbidity']
values = [3.716080 	,129.422921, 	18630.057858 ,	332.615625, 	592.885359, 	4.500656 	]


water_sample = dict(zip(features, values))


def make_prediction(water_sample):
    # Convert the water sample into a DataFrame and then to DMatrix
    df_sample = pd.DataFrame([water_sample])  # Convert to DataFrame (one row of data)
    dmatrix_sample = xgb.DMatrix(df_sample, feature_names=features)  # Create DMatrix
    prediction = model.predict(dmatrix_sample)  # Make prediction
    return prediction


y_pred=make_prediction(water_sample)[0]


if y_pred <0.5 :
    print('0, water is NOT potable')
else:
    print('1, water is potable')


