import comet_ml
from comet_ml.integration.sklearn import log_model
from  sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import optuna
from pathlib import Path
import shutil
import pandas as pd
from data_preprocessing import get_preprocessing_pipeline, perform_feature_engineering

comet_ml.login()

preprocessing_pipeline = get_preprocessing_pipeline()

artifact_path = Path("../artifacts")
if not artifact_path.exists():
    artifact_path.mkdir(exist_ok=True)

def objective(trial):
    experiment = comet_ml.start(project_name="HVP")
    experiment.add_tag("tuning")

    artifact = experiment.get_artifact("California_Houses", version_or_alias="latest")

    artifact.download(artifact_path, overwrite_strategy="PRESERVE")

    df = pd.read_csv(artifact_path.joinpath(f"{artifact.name}.csv"))

    dataset = perform_feature_engineering(df)

    X = dataset.drop(columns=["Median_House_Value"], inplace=False)
    Y = dataset["Median_House_Value"]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "criterion": trial.suggest_categorical("criterion", ["squared_error", "absolute_error", "friedman_mse", "poisson"]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
        "random_state": 42,
        "n_jobs": -1
    }

    model = make_pipeline(preprocessing_pipeline, RandomForestRegressor(**params))

    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    rmse = root_mean_squared_error(Y_test, Y_pred)
    mae = mean_absolute_error(Y_test, Y_pred)
    mape = mean_absolute_percentage_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)

    experiment.log_parameters(parameters=params)
    experiment.log_metric("rmse", rmse)
    experiment.log_metric("mae", mae)
    experiment.log_metric("mape", mape)
    experiment.log_metric("r2", r2)

    log_model(experiment=experiment, model_name="HVP", model=model)

    experiment.end()

    return rmse, mae, mape, r2

study = optuna.create_study(directions=["minimize", "minimize", "minimize", "maximize"])
study.optimize(objective, n_trials=50)

shutil.rmtree(artifact_path)