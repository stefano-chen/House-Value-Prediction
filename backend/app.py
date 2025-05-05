import sys

from flask import Flask, request
import comet_ml
import joblib
from pathlib import Path
import pandas as pd
import numpy as np

def calculate_confidence(model, x):
    tree_preds = np.array([tree.predict(model.named_steps["pipeline"].transform(x)) for tree in model.named_steps["randomforestregressor"].estimators_])

    y_pred = tree_preds.mean(axis=0)[0]

    return max(0, 100 * (1- (tree_preds.std(axis=0)[0] / y_pred))) if y_pred !=0 else 0.0

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
        return {
            "prediction": round(model.predict(x)[0],2),
            "confidence": round(calculate_confidence(model,x), 2)
        }, 200

    return app