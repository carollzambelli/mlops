"""
Transformes utils class to be used in pipeline and data schema validation
"""

import pandas as pd
import numpy as np
import pickle
from pydantic import ValidationError
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV                                                
from sklearn.linear_model import LinearRegression, Ridge
from scipy.stats import loguniform 
from datetime import datetime
from src.core import config, MultipleDataSchema, ASSETS_PATH


def validate_inputs(raw_data: pd.DataFrame):
    """Validade columns and data type follow the model requirements """

    errors = None
    features = config.data_config.quali_variables + config.data_config.quanti_variables
    data = raw_data[features+ [config.ml_config.target]]

    try:
        MultipleDataSchema(inputs=data.replace({np.nan: None}).to_dict(orient="records"))
    except ValidationError as error:
        errors = error.json()
    
    return data, errors


class Wrapper(BaseEstimator):
    def __init__(self, intermediate_model, name_model):                
        self.intermediate_model = intermediate_model
        self.name_model = name_model
    def fit(self, X, y=None):                        
        return self                                  
    def transform(self, X_teste):
        dt = datetime.now().date()
        filename = f'{ASSETS_PATH}/{config.ml_config.trained_model_file}_{self.name_model}_{dt}.pkl'
        pickle.dump(self.intermediate_model, open(filename, 'wb'))
        return self.intermediate_model.predict(X_teste)


def train_pipe(X_treino, y_treino):

    preprocessador = ColumnTransformer(transformers=[
    ("quanti", StandardScaler(), config.data_config.quanti_variables),
    ("quali", OneHotEncoder(
        sparse_output=False, drop="first", handle_unknown='ignore'
        ),config.data_config.quali_variables )])
    
    RL = Pipeline([
            ("preprocess", preprocessador),
            ("linear regression", RandomizedSearchCV(estimator=LinearRegression(),
                    param_distributions={},
                    scoring='neg_root_mean_squared_log_error',
                    cv=10) )
                    ])

    RD =  Pipeline([
        ("preprocess", preprocessador),
        ("linear regression", RandomizedSearchCV(estimator=Ridge(),
                  param_distributions={'alpha': loguniform(1e-5, 1e1)},
                  scoring='neg_root_mean_squared_log_error',
                  cv=10) )
                  ])
    
    treino_predict_pipe = FeatureUnion([
        ('linear', Wrapper(RL.fit(X_treino, y_treino), "linear")),
        ('ridge', Wrapper(RD.fit(X_treino, y_treino), "ridge"))
        ])
    
    return treino_predict_pipe

