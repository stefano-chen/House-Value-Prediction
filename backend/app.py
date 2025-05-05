import sys

from flask import Flask, request
import comet_ml
import joblib
from pathlib import Path
import pandas as pd
import numpy as np

def calculate_confidence(model, x):
    X_transformed = model.named_steps["pipeline"].transform(x)

    tree_preds = np.array([tree.predict(X_transformed) for tree in model.named_steps["randomforestregressor"].estimators_])

    y_pred = tree_preds.mean(axis=0)[0]
    y_std = tree_preds.std(axis=0)[0]

    return max(0, 100 * (1- (y_std / y_pred))) if y_pred !=0 else 0.0

def download_model():
    comet_ml.login()

    model_registry = comet_ml.API().get_model(
        workspace=comet_ml.API().get_default_workspace(), model_name="HVP"
    )

    versions = model_registry.find_versions(status="Production")

    if len(versions) == 0:
        print("No model Found", file=sys.stderr)
        sys.exit(-1)

    model_registry.download(versions[0], output_folder=Path("./backend/model"))

    model = joblib.load(Path("./backend/model/model-data/comet-sklearn-model.joblib"))

    return model, versions[0]


def create_app():

    model, version = download_model()

    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health_check():
        return {}, 200

    @app.route("/version", methods=["GET"])
    def get_model_version():
        return {"version": version}, 200

    @app.route("/predict", methods=["POST"])
    def get_model_prediction():
        x = pd.DataFrame([request.form])
        prediction = model.predict(x)
        confidence = calculate_confidence(model, x)
        return {
            "prediction": round(prediction[0],2),
            "confidence": round(confidence, 2)
        }, 200


    return app