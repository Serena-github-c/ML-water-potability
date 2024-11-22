import pickle
import pandas as pd
import xgboost as xgb
from flask import Flask, request, jsonify

model_file='water_model.bin.'
features = ['ph', 'hardness', 'solids', 'sulfate', 'conductivity',
       'turbidity']

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)


app=Flask('potability')

@app.route('/predict', methods=['POST'])
def predict():
    water_sample= request.get_json()

    X_sample = pd.DataFrame([water_sample])
    X = xgb.DMatrix(X_sample, feature_names=features)
    y_pred = model.predict(X)[0]
    ans = y_pred > 0.5
    
    result = { 'probability' : float(y_pred),
                'answer': bool(ans)
                }

    return jsonify(result)
        
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
    
    
