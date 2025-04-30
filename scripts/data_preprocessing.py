from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
import numpy as np
from sklearn.pipeline import Pipeline


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