from flask import Flask
import comet_ml
import joblib
from pathlib import Path

def create_app():

    comet_ml.login()

    api = comet_ml.API()

    model_registry = api.get_model(
        workspace=api.get_default_workspace(), model_name="HVP"
    )

    latest_version = model_registry.find_versions(status="Production")[0]

    model_registry.download(latest_version, output_folder=Path("./backend/model"))

    model = joblib.load(Path("./backend/model/model-data/comet-sklearn-model.joblib"))

    app = Flask(__name__)

    app.route("/health")
    def health_check():
        return {}, 200

    return app