import sys
import tempfile
from pathlib import Path

import comet_ml
import joblib
import numpy as np
import pandas


class Model:

    def __init__(self, api_key):
        comet_ml.login(api_key=api_key)
        self._versions = None
        self._model = None

    def download_model(self, model_name):

        try:
            model_registry = comet_ml.API().get_model(
                workspace=comet_ml.API().get_default_workspace(), model_name=model_name
            )
        except ValueError:
            print("Missing COMET_API_KEY environment variable", file=sys.stderr)
            sys.exit(-1)

        self._versions = model_registry.find_versions(status="Production")

        if len(self._versions) == 0:
            print("No model Found", file=sys.stderr)
            raise ValueError("Invalid model name")

        with tempfile.TemporaryDirectory() as tempdir:
            tempdir_path = Path(tempdir)
            model_registry.download(self._versions[0], output_folder=tempdir_path)
            self._model = joblib.load(tempdir_path.joinpath("model-data", "comet-sklearn-model.joblib"))

    def _calculate_confidence(self, x):
        x_tranformed = self._model.named_steps["pipeline"].transform(x)
        tree_preds = np.array(
            [tree.predict(x_tranformed) for tree in self._model.named_steps["randomforestregressor"].estimators_])

        y_pred = tree_preds.mean(axis=0)[0]

        return max(0, 100 * (1 - (tree_preds.std(axis=0)[0] / y_pred))) if y_pred != 0 else 0.0

    def get_version(self):
        if self._model is None:
            return -1
        return self._versions[0]

    def predict(self, x: pandas.DataFrame):
        if self._model is None:
            raise RuntimeError("model not available for prediction")
        prediction = round(self._model.predict(x)[0], 2)
        confidence = round(self._calculate_confidence(x), 2)

        return prediction, confidence
