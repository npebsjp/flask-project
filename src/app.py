from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import sqlite3
import pickle
from pickle import load
import os

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# = os.path.join(BASE_DIR, "../models/model_columns.csv")

# carregando as colunas esperadas
expected_cols = pd.read_csv("../models/model_columns.csv")["column"].tolist()
model_path = os.path.join(BASE_DIR, "../models/ligthGBM_algorithm_regressor_default_42.sav")

# Load trained model
model = pickle.load(open("../models/ligthGBM_algorithm_regressor_default_42.sav", "rb"))



# Load the expected columns from training (assume saved from training step)
#expected_cols = pd.read_csv("models/model_columns.csv")["column"].tolist()  # Save this during training


@app.route("/", methods=["GET", "POST"])
def index():
    predicted_price = None
    if request.method == "POST":
        # Get inputs from form
        user_input = {
            "zipCode": request.form["zipCode"],
            #"latitude": float(request.form["latitude"]),
            #"longitude": float(request.form["longitude"]),
            "bedrooms": int(request.form["bedrooms"]),
            "bathrooms": float(request.form["bathrooms"]),
            "squareFootage": int(request.form["squareFootage"]),
            "lotSize": int(request.form["lotSize"]),
            "yearBuilt": int(request.form["yearBuilt"]),
            "coolingType": request.form["coolingType"],
            "heatingType": request.form["heatingType"],
            "pool": request.form["pool"],
            "garageSpaces": request.form["garageSpaces"],
            "propertyType": request.form["propertyType"],
            "city": request.form["city"],
            "county": request.form["county"]
        }

        # Start with basic features
        input_df = pd.DataFrame([{
            "zipCode": user_input["zipCode"],
            #"latitude": user_input["latitude"],
            #"longitude": user_input["longitude"],
            "bedrooms": user_input["bedrooms"],
            "bathrooms": user_input["bathrooms"],
            "squareFootage": user_input["squareFootage"],
            "lotSize": user_input["lotSize"],
            "yearBuilt": user_input["yearBuilt"]
        }])

        # One-hot encode categorical values
        for prefix in ["coolingType", "heatingType", "pool", "propertyType", "garageSpaces", "city", "county"]:
            col_name = f"{prefix}_{user_input[prefix]}"
            input_df[col_name] = 1

        # Add missing expected columns
        for col in expected_cols:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder to match model
        input_df = input_df[expected_cols]

        # Make prediction
        predicted_price = model.predict(input_df)[0]

    return render_template("index.html", predicted_price=predicted_price)
