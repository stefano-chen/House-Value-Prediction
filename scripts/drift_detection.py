import os
import sys
import tempfile
from pathlib import Path
import comet_ml
import pandas as pd
from evidently.presets import DataDriftPreset
from evidently import Report
from pymongo.mongo_client import MongoClient, ConnectionFailure

try:
    mongo_uri = os.environ["MONGODB_URI"]
except KeyError:
    print("missing MONGODB_URI env variable", file=sys.stderr)
    sys.exit(-1)

comet_ml.login()

client = MongoClient(mongo_uri)

try:
    # The ping command is cheap and does not require auth.
    client.admin.command('ping')
except ConnectionFailure:
    print("Connection Error", file=sys.stderr)
    sys.exit(-1)

documents = list(client['HVP']["predictions"].find())

if len(documents) <=0:
    print("no data in the database", file=sys.stderr)
    sys.exit(-1)

new_data = pd.DataFrame(documents)
new_data.drop(columns=["_id", "Prediction", "Confidence", "Model_Version", "Date"], inplace=True, errors="ignore")

experiment = comet_ml.start(project_name="HVP")
experiment.add_tag("drift_check")


with tempfile.TemporaryDirectory() as tempdir:
    tempdir_path = Path(tempdir)
    artifact = experiment.get_artifact("California_Houses")
    artifact.download(tempdir_path, overwrite_strategy="PRESERVE")
    reference_data = pd.read_csv(tempdir_path.joinpath(f"{artifact.assets[0].logical_path}")).drop(columns=["Median_House_Value"], inplace=False)

data_drift_report = Report(metrics=[DataDriftPreset()])

result = data_drift_report.run(reference_data=reference_data, current_data=new_data).dict()

n_drifted_col = result["metrics"][0]["value"]["count"]

if n_drifted_col > 0:
    print("Data drift detected. Retraining the model", file=sys.stdout)
    with open("drift_detected.txt", "w") as f:
        f.write("drift_detected")
else:
    print("No data drift detected", file=sys.stdout)
    with open("drift_detected.txt", "w") as f:
        f.write("no_drift")