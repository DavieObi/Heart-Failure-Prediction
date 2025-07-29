# app.py

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('xgboost_heart_failure_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract features
    age = data['age']
    anaemia = data['anaemia']
    cpk = data['cpk']
    diabetes = data['diabetes']
    ejection = data['ejection']
    bp = data['bp']
    platelets = data['platelets']
    creatinine = data['creatinine']
    sodium = data['sodium']
    sex = data['sex']
    smoking = data['smoking']

    # Compute engineered features
    creatinine_sodium_ratio = creatinine / sodium
    cpk_platelet_ratio = cpk / platelets
    anaemia_diabetes = anaemia * diabetes
    bp_smoking = bp * smoking
    age_ejection_interaction = age * ejection

    # Assemble all features (same order as training)
    features = np.array([
        age, anaemia, cpk, diabetes, ejection, bp,
        platelets, creatinine, sodium, sex, smoking,
        creatinine_sodium_ratio, cpk_platelet_ratio,
        anaemia_diabetes, bp_smoking, age_ejection_interaction
    ]).reshape(1, -1)

    # Scale features
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)[0]
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)