import comet_ml
from comet_ml.integration.sklearn import log_model
from  sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from pathlib import Path
import shutil
import pandas as pd
from data_preprocessing import get_preprocessing_pipeline, perform_feature_engineering


comet_ml.login(project_name="HVP")

model_params = {
    "n_estimators": {
        "type": "integer",
        "scaling_type": "uniform",
        "min": 100,
        "max": 300
    },
    "criterion": {
        "type": "categorical",
        "values": ["squared_error", "absolute_error", "friedman_mse", "poisson"]
    },
    "min_samples_leaf": {
        "type": "discrete",
        "values": [1, 3, 5, 7, 9]
    }
}

spec = {
    "objective": "minimize",
    "metric": "rmse"
}

optimizer_config = {
    "algorithm": "bayes",
    "spec": spec,
    "parameters": model_params,
    "name": "Bayes Optimization",
}

opt = comet_ml.Optimizer(config=optimizer_config)

preprocessing_pipeline = get_preprocessing_pipeline()

artifact_path = Path("../artifacts")
if not artifact_path.exists():
    artifact_path.mkdir(exist_ok=True)

for experiment in opt.get_experiments():
    experiment.add_tag("train")

    artifact = experiment.get_artifact("California_Houses", version_or_alias="latest")

    artifact.download(artifact_path, overwrite_strategy="PRESERVE")

    df = pd.read_csv(artifact_path.joinpath(f"{artifact.name}.csv"))

    dataset = perform_feature_engineering(df)

    X = dataset.drop(columns=["Median_House_Value"], inplace=False)
    Y = dataset["Median_House_Value"]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)

    params = {
        "n_estimators": experiment.get_parameter("n_estimators"),
        "criterion": experiment.get_parameter("criterion"),
        "min_samples_leaf": experiment.get_parameter("min_samples_leaf"),
        "random_state": 42,
        "n_jobs":-1
    }

    model = make_pipeline(preprocessing_pipeline, RandomForestRegressor(**params))

    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    experiment.log_parameters(parameters=params)
    experiment.log_metric("rmse", root_mean_squared_error(Y_test, Y_pred))
    experiment.log_metric("mae", mean_absolute_error(Y_test, Y_pred))
    experiment.log_metric("mape", mean_absolute_percentage_error(Y_test, Y_pred))

    log_model(experiment=experiment, model_name="HVP", model=model)

    experiment.end()

shutil.rmtree(artifact_path)