import sys

from flask import Flask, request
import comet_ml
import joblib
from pathlib import Path
import pandas as pd
import numpy as np

def calculate_confidence(model, x):
    preprocessing = model.named_steps["pipeline"]
    forest = model.named_steps["randomforestregressor"]

    X_transformed = preprocessing.transform(x)

    tree_preds = np.array([tree.predict(X_transformed) for tree in forest.estimators_])

    y_pred = tree_preds.mean(axis=0)[0]
    y_std = tree_preds.std(axis=0)[0]

    return max(0, 100 * (1- (y_std / y_pred))) if y_pred !=0 else 0.0

def download_model():
    comet_ml.login()

    api = comet_ml.API()

    model_registry = api.get_model(
        workspace=api.get_default_workspace(), model_name="HVP"
    )

    versions = model_registry.find_versions(status="Production")

    if len(versions) == 0:
        print("No model Found", file=sys.stderr)
        sys.exit(-1)

    latest_version = versions[0]

    model_registry.download(latest_version, output_folder=Path("./backend/model"))

    model = joblib.load(Path("./backend/model/model-data/comet-sklearn-model.joblib"))

    return model, latest_version


def create_app(testing=False):

    model, version = download_model()

    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health_check():
        return {}, 200

    @app.route("/version", methods=["GET"])
    def get_model_version():
        return {"version":version}, 200

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