from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
import numpy as np
from sklearn.pipeline import Pipeline

import sys
from pathlib import Path
import pandas as pd

def perform_feature_engineering(df: pd.DataFrame):

    df['Rooms_Per_House'] = df['Tot_Rooms'] / df['Households']
    df['Bedrooms_Ratio'] = df['Tot_Bedrooms'] / df['Tot_Rooms']
    df['People_Per_House'] = df['Population'] / df['Households']
    df.drop(columns=["Tot_Rooms", "Tot_Bedrooms"], inplace=True)

    return df

def get_preprocessing_pipeline():
    log_transformer = FunctionTransformer(np.log, feature_names_out="one-to-one")

    preprocessing_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="mean").set_output(transform="pandas")),
        ("column transformer", ColumnTransformer([
            ("log transformation", log_transformer, ["Population", "Households", "Median_Income"])
        ], remainder='passthrough')),
        ("standard scaler", StandardScaler())
    ])

    return preprocessing_pipeline


if __name__=="__main__":
    if len(sys.argv) < 3:
        print("missing arguments", file=sys.stderr)
        sys.exit(-1)

    dataset_path = Path(sys.argv[1])

    if not dataset_path.exists():
        print("File not Found", file=sys.stderr)
        sys.exit(-1)

    save_path = Path(sys.argv[2])

    df = pd.read_csv(dataset_path)

    df = perform_feature_engineering(df)

    df.to_csv(save_path, index=False)