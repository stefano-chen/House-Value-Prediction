import sys
import comet_ml
import joblib
from pathlib import Path
import numpy as np
import tempfile

class Model:

    def __init__(self):
        comet_ml.login()
        self.versions = None
        self.model = None
        self.model_registry = None

    def download_model(self):

        self.model_registry = comet_ml.API().get_model(
            workspace=comet_ml.API().get_default_workspace(), model_name="HVP"
        )

        self.versions = self.model_registry.find_versions(status="Production")

        if len(self.versions) == 0:
            print("No model Found", file=sys.stderr)
            sys.exit(-1)

        with tempfile.TemporaryDirectory() as tempdir:
            tempdir_path = Path(tempdir)
            self.model_registry.download(self.versions[0], output_folder=tempdir_path)
            self.model = joblib.load(tempdir_path.joinpath("model-data", "comet-sklearn-model.joblib"))

    def _calculate_confidence(self, x):
        x_tranformed = self.model.named_steps["pipeline"].transform(x)
        tree_preds = np.array(
            [tree.predict(x_tranformed) for tree in self.model.named_steps["randomforestregressor"].estimators_])

        y_pred = tree_preds.mean(axis=0)[0]

        return max(0, 100 * (1 - (tree_preds.std(axis=0)[0] / y_pred))) if y_pred != 0 else 0.0

    def get_version(self):
        if self.model is None:
            return -1
        return self.versions[0]

    def predict(self, x):
        if self.model is None:
            return -1,-1
        prediction = round(self.model.predict(x)[0],2)
        confidence = round(self._calculate_confidence(x),2)

        return prediction, confidence