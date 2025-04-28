from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.pipeline import Pipeline
import joblib

dataset_path = "../dataset/processed/California_Houses_Processed_Train.csv"
imputer_path = "../artifacts/preprocessing/imputer.pkl"
scaler_path = "../artifacts/preprocessing/scaler.pkl"
preprocessing_pipeline_path = "../artifacts/preprocessing/preprocessing_pipeline.pkl"

houses_df = pd.read_csv(dataset_path)
houses_df.drop(columns=["Median_House_Value"], inplace=True)

preprocessing_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="mean")),
    ("standard scaler", StandardScaler())
])

preprocessing_pipeline.fit(houses_df)

imputer = preprocessing_pipeline.named_steps["impute"]
scaler = preprocessing_pipeline.named_steps["standard scaler"]

joblib.dump(imputer, imputer_path)
joblib.dump(scaler, scaler_path)
joblib.dump(preprocessing_pipeline, preprocessing_pipeline_path)

