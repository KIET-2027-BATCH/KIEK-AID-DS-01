import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__, template_folder='templates')

# Load trained models and scaler
try:
    rf_model = joblib.load("random_forest_model.pkl")
    xgb_model = joblib.load("xgboost_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError as e:
    print(f"Model file missing: {e}")
    rf_model, xgb_model, scaler = None, None, None

# Load feature names or extract from model
feature_names = None
try:
    feature_names = joblib.load("feature_names.pkl")
except FileNotFoundError:
    print("Feature names file is missing. Extracting from model...")
    if rf_model:
        feature_names = rf_model.feature_names_in_.tolist()  # Extract from trained model
    elif xgb_model:
        feature_names = xgb_model.feature_names_in_.tolist()

@app.route('/')
def home():
    return render_template('index.html', prediction_text=None)  # No message initially

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if rf_model is None or xgb_model is None:
            return render_template('index.html', prediction_text="Error: Model files are missing!")

        # Collect input data
        input_data = {
            "Kilometers_Driven": float(request.form['Kilometers_Driven']),
            "Mileage": float(request.form['Mileage']),
            "Engine": float(request.form['Engine']),
            "Power": float(request.form['Power']),
            "Seats": int(request.form['Seats']),
            "Fuel_Type_Diesel": 1 if request.form['Fuel_Type'] == "Diesel" else 0,
            "Fuel_Type_Petrol": 1 if request.form['Fuel_Type'] == "Petrol" else 0,
            "Fuel_Type_CNG": 1 if request.form['Fuel_Type'] == "CNG" else 0,
            "Transmission_Manual": 1 if request.form['Transmission'] == "Manual" else 0,
            "Owner_Type_First": 1 if request.form['Owner_Type'] == "First" else 0
        }

        # One-hot encode locations
        locations = ["Bangalore", "Chennai", "Coimbatore", "Delhi", "Hyderabad", "Kolkata", "Mumbai", "Pune"]
        for loc in locations:
            input_data[f"Location_{loc}"] = 1 if request.form['Location'] == loc else 0

        # Convert to DataFrame
        df_input = pd.DataFrame([input_data])

        # Ensure all expected features exist (set missing ones to 0)
        if feature_names:
            for feature in feature_names:
                if feature not in df_input.columns:
                    df_input[feature] = 0

            # Reorder columns to match training order
            df_input = df_input[feature_names]
        else:
            return render_template('index.html', prediction_text="Error: Feature names cannot be determined!")

        # Ensure scaler is loaded before transforming data
        if scaler:
            numerical_features = ["Kilometers_Driven", "Mileage", "Engine", "Power", "Seats"]
            df_input[numerical_features] = scaler.transform(df_input[numerical_features])
        else:
            return render_template('index.html', prediction_text="Error: Scaler file is missing!")

        # Predict car price
        price_rf = rf_model.predict(df_input)[0]
        price_xgb = xgb_model.predict(df_input)[0]
        predicted_price = (price_rf + price_xgb) / 2  # Averaging both models

        return render_template('index.html', prediction_text=f'Estimated Car Price: ₹{predicted_price:,.2f}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
