from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path

dataset_path = Path("../dataset/processed/California_Houses_Processed_Train.csv")
preprocessing_pipeline_path = Path("../artifacts/preprocessing/preprocessing_pipeline.pkl")

houses_df = pd.read_csv(dataset_path)
houses_df.drop(columns=["Median_House_Value"], inplace=True)

log_transformer = FunctionTransformer(np.log, feature_names_out="one-to-one")

preprocessing_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="mean").set_output(transform="pandas")),
    ("column transformer", ColumnTransformer([
        ("log transformation", log_transformer, ["Population", "Households", "Median_Income"])
    ], remainder='passthrough')),
    ("standard scaler", StandardScaler())
])

preprocessing_pipeline.fit(houses_df)

joblib.dump(preprocessing_pipeline, preprocessing_pipeline_path)

