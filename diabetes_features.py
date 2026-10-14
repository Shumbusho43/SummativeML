"""Feature definitions shared by the notebook and the saved model pipelines."""
import numpy as np
import pandas as pd

BINARY_COLS = ["HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack",
               "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
               "NoDocbcCost", "DiffWalk", "Sex"]
ORDINAL_COLS = ["GenHlth", "Age", "Education", "Income"]
CONT_COLS = ["BMI", "MentHlth", "PhysHlth"]
RAW_FEATURES = BINARY_COLS + ORDINAL_COLS + CONT_COLS

ENG_NUMERIC = ["CardioMetabolicLoad", "LifestyleScore", "UnhealthyDays", "BMI_Category",
               "AgeBMI", "SES_Index"]
ENG_BINARY = ["CareBarrier", "PoorGenHlth", "Obese"]


def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """Row-wise, parameter-free feature engineering (safe to apply before or after splitting)."""
    X = X.copy()
    X["BMI"] = X["BMI"].clip(12, 60)
    X["CardioMetabolicLoad"] = X[["HighBP", "HighChol", "Stroke", "HeartDiseaseorAttack"]].sum(axis=1)
    X["LifestyleScore"] = (X[["PhysActivity", "Fruits", "Veggies"]].sum(axis=1)
                           - X[["Smoker", "HvyAlcoholConsump"]].sum(axis=1))
    X["UnhealthyDays"] = X["MentHlth"] + X["PhysHlth"]
    X["BMI_Category"] = np.digitize(X["BMI"], [18.5, 25, 30, 35, 40])
    X["AgeBMI"] = X["Age"] * X["BMI"]
    X["SES_Index"] = X["Education"] + X["Income"]
    X["CareBarrier"] = ((X["AnyHealthcare"] == 0) | (X["NoDocbcCost"] == 1)).astype(int)
    X["PoorGenHlth"] = (X["GenHlth"] >= 4).astype(int)
    X["Obese"] = (X["BMI"] >= 30).astype(int)
    return X
