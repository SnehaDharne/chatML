
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import json

MODEL_REGISTRY = {
    "decision_tree": {
        "class": DecisionTreeClassifier,
        "defaults": {
            "criterion": "gini",
            "splitter": "best",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "min_weight_fraction_leaf": 0.0,
            "max_features": None,
            "random_state": None,
            "max_leaf_nodes": None,
            "min_impurity_decrease": 0.0,
            "class_weight": None,
            "ccp_alpha": 0.0
        }
    },
    "svm": {
        "class": SVC,
        "defaults": {
            "C": 1.0,
            "kernel": "rbf",
            "degree": 3,
            "gamma": "scale",
            "coef0": 0.0,
            "shrinking": True,
            "probability": False,
            "tol": 0.001,
            "cache_size": 200,
            "class_weight": None,
            "verbose": False,
            "max_iter": -1,
            "decision_function_shape": "ovr",
            "break_ties": False,
            "random_state": None
        }
    },
    "logistic_regressioin": {
        "class": LogisticRegression,
        "defaults": {
            "penalty": 'l2',
            "dual": False,
            "tol": 0.0001,
            "C": 1.0,
            "fit_intercept": True,
            "intercept_scaling": 1,
            "class_weight": None,
            "random_state": None,
            "solver": 'lbfgs',
            "max_iter": 100,
            "multi_class": 'auto',
            "verbose": 0,
            "warm_start": False,
            "n_jobs": None,
            "l1_ratio": None
        }
    }
}

def preprocess_data(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.fillna(df.mean(numeric_only=True), inplace=True)
    return df

def run_model_from_json(json_obj: dict):
    config = json_obj

    model_name = config['model_name']
    filename = config['filename']
    target = config['target_variable']
    split = config.get('split', 0.2)
    user_params = config.get('param', {})

    df = preprocess_data(filename)

    if target not in df.columns:
        raise ValueError(f"Target variable '{target}' not found in dataset.")

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=42)

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' is not supported.")

    model_info = MODEL_REGISTRY[model_name]
    model_class = model_info['class']
    default_params = model_info['defaults']

    combined_params = {**default_params, **user_params}
    model = model_class(**combined_params)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    classification_report_rep = classification_report(y_test, y_pred)
    return {
        "accuracy": accuracy,
        "classification_report" : classification_report_rep,
        "params_used": combined_params
    }
