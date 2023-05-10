"""
This Module contains a variety of different model defintions (functions)
"""
from typing import Any
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import pandas as pd



def iforest(X_train: pd.DataFrame,
            y_train: pd.DataFrame,
            contamination: float = None,
            n_estimator: int = 150,
            max_feature: int = 8,
            bootstrap: bool = True) -> Any:
    """
    Trains a model given data. Logs metrics to the run context.

    :param X_train: Input variables for training
    :param y_train: target variable. y_train will not be used to fit Isolation Forest
    but to calculate the contamination parameter.
    :return: a trained model
    """
    # Normalize data
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(X_train)
    X_train = min_max_scaler.transform(X_train)
    if not contamination:
        contamination = round(y_train.value_counts()[-1]/y_train.value_counts()[1], 3)
    model = IsolationForest(n_estimators=n_estimator,
                            contamination=contamination,
                            max_features=max_feature,
                            bootstrap=bootstrap, n_jobs=-1)
    model.fit(X_train)
    return model
