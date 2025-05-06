import os
import sys

from flask import Flask, request

import pandas as pd

from pymongo.mongo_client import MongoClient

from backend.classes.Model import Model

def extract_feature_from_request(body):
    features_names = ["Median_Income", "Median_Age", "Population", "Households", "Latitude", "Longitude",
                      "Distance_to_coast", "Distance_to_LA", "Distance_to_SanDiego", "Distance_to_SanJose",
                      "Distance_to_SanFrancisco", "Rooms_Per_House", "Bedrooms_Ratio", "People_Per_House"]
    x = {}
    for feature_name in features_names:
        x[feature_name] = body[feature_name] if feature_name in body else None

    return pd.DataFrame([x]), x

def create_app(testing=False):

    # mongo_uri = os.environ["MONGODB_URI"]
    #
    # mongo_client = MongoClient(mongo_uri)
    # mongo_db = mongo_client["HVP"]
    # mongo_collection = mongo_db["Houses"]

    model = Model()
    if not testing:
        model.download_model()

    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health_check():
        return {}, 200

    @app.route("/version", methods=["GET"])
    def get_model_version():
        return {"version": model.get_version()}, 200

    @app.route("/predict", methods=["POST"])
    def get_model_prediction():
        x_df, x_dict = extract_feature_from_request(request.form)
        x_dict["Prediction"], x_dict["Confidence"] = model.predict(x_df)
        # mongo_collection.insert_one(x)
        return {
            "prediction": x_dict["Prediction"],
            "confidence": x_dict["Confidence"]
        }, 200

    return app