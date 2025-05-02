import comet_ml
from pathlib import Path

dataset_path = Path("../dataset/raw/California_Houses.csv")

comet_ml.login()

artifact = comet_ml.Artifact(
    name="California_Houses",
    artifact_type="dataset",
    aliases=["raw"],
    metadata={"task": "regression"}
)

artifact.add(dataset_path)

experiment = comet_ml.start(project_name="HVP")
experiment.add_tag("upload")
experiment.log_artifact(artifact)

experiment.end()