import sys

import comet_ml
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="this scripts is used to versioning a dataset to CometML, if the artifact name is already in use, the remote artifact will be overwrite")
parser.add_argument("path", help="path of the dataset to versioning")
parser.add_argument("artifact_name", help="the artifact name, if the artifact already exists it will be overwritten")
parser.add_argument("-p","--project_name", help="the project name, if does not exists it will create a new one with the given name if not provided the experiment will be saved in the Uncategorized Experiments")
args = parser.parse_args()


dataset_path = Path(args.path)

if not dataset_path.exists():
    print("Dataset path does not map to a file", file=sys.stderr)
    sys.exit(-1)

comet_ml.login()

artifact = comet_ml.Artifact(
    name=args.artifact_name,
    artifact_type="dataset",
    metadata={"task": "regression"}
)

artifact.add(dataset_path)

project_name = args.project_name if args.project_name else ""

experiment = comet_ml.start(project_name=project_name)
experiment.add_tag("upload")
experiment.log_artifact(artifact)

experiment.end()