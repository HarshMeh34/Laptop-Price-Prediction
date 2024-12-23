

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os

df = pd.read_csv('i:/Machine Learning Krish/LaptopPrice/flask_LaptopPrice/output1.csv')



app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from form
        company = request.form['Company']
        typename = request.form['TypeName']
        cpu = request.form['Cpu']
        ram = int(request.form['Ram'])
        gpu = request.form['Gpu']
        weight = float(request.form['Weight'])
        ssd = int(request.form['SSD'])
        hdd = int(request.form['HDD'])
        ppi = float(request.form['ppi'])
        os = request.form['os']

        # Create a DataFrame with user input
        user_input = pd.DataFrame([[company, typename, cpu, gpu, os, ram, weight, ssd, hdd, ppi]],
                                    columns=['Company', 'TypeName', 'Cpu', 'Gpu', 'os', 'Ram', 'Weight', 'SSD', 'HDD', 'ppi'])
        user_input['Ram']= round(np.log(user_input['Ram']),2)
        user_input['ppi']= round(np.log(user_input['ppi']),4)
        # Preprocess user input
        categorical_features = ['Company', 'TypeName', 'Cpu', 'Gpu', 'os']
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(df[categorical_features])  # Assuming 'df' is your training data

        user_input_encoded = encoder.transform(user_input[categorical_features])
        feature_names = encoder.get_feature_names_out()
        user_input_encoded = pd.DataFrame(user_input_encoded, columns=feature_names)

        # Combine encoded categorical features with numerical features
        user_input_encoded = pd.concat([user_input_encoded, user_input[['Ram', 'Weight', 'SSD', 'HDD', 'ppi']]], axis=1)

        # Make prediction
        prediction = model.predict(user_input_encoded)[0]
        # original_value = np.exp(log_value)
        prediction_price = np.exp(prediction)
        return render_template('index.html', prediction=prediction_price)


if __name__ == '__main__':
    app.run(debug=True)