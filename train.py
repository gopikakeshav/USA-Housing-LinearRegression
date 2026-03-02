import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# onehotencoder # standardscaler # simpleimputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# model # train-test split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import joblib

def train_model():

    df = pd.read_csv("data/USA_Housing.csv")

    X = df.drop(columns = ["Price", "Address"])
    y = df["Price"]

    numerical_cols = X.select_dtypes(include = ["int64", "float64"]).columns

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42) 

    # Numerical
    numerical_transformer = Pipeline([
                            ("impute", SimpleImputer(strategy = "mean")),
                            ("scaler", StandardScaler())
                            ])

    # dataprep
    data_prep = ColumnTransformer(transformers = [
                        ("numerical", numerical_transformer, numerical_cols)
                        ])
    # model
    LR = LinearRegression()

    # final pipeline
    model_pipeline = Pipeline([
                            ("impute_scaling", data_prep),
                            ("modelling_lr", LR)
                            ])


    model_pipeline.fit(X_train, y_train)
    print("Test R2 : ", model_pipeline.score(X_test, y_test))

    joblib.dump(model_pipeline, "models/house_price_lr.pkl")

if __name__ == "__main__":
    print("Training Model...")
    train_model()
    