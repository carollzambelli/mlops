"""
Train script following:
- core configuration
- pipeline data transformation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import logging
from scipy.stats import loguniform   
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.metrics import r2_score, root_mean_squared_error 
from datetime import datetime
from src.core import config, PACKAGE_ROOT, ASSETS_PATH
from src.utils import validate_inputs

logging.basicConfig(level=logging.INFO)

def train():
    logging.info("Iniciando treinamento")
    data = pd.read_csv(f'{PACKAGE_ROOT}/{config.ml_config.train_data_path}', index_col=0)
    validated_data, errors = validate_inputs(raw_data=data, step = "train")

    if not errors:

        y = data[config.ml_config.target]
        features = config.data_config.quali_variables + config.data_config.quanti_variables
        X = data[features]

        X_treino, X_teste, y_treino, y_teste = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=123)

        preprocessador = ColumnTransformer(transformers=[
        ("quanti", StandardScaler(), config.data_config.quanti_variables),
        ("quali", OneHotEncoder(sparse_output=False, drop="first", handle_unknown='ignore'), config.data_config.quanli_variables )])

        RD =  Pipeline([
        ("preprocess", preprocessador),
        ("linear regression", RandomizedSearchCV(estimator=Ridge(),
                  param_distributions={'alpha': loguniform(1e-5, 1e1) },
                  scoring='neg_root_mean_squared_log_error',
                  cv=10) )])
        
        clf  = RD.fit(X_treino, y_treino)
        y_hat = clf.predict(preprocessador.fit_transform(X_teste))
        erro = root_mean_squared_error(y_teste, y_hat)

    
    if  erro <= config.ml_config.erro :
        dt = datetime.now().date()
        filename = f'{PACKAGE_ROOT}/{config.ml_config.trained_model_file}{dt}.pkl'
        pickle.dump(clf, open(filename, 'wb'))
        logging.info(f"Modelo {filename} salvo, com score {erro}")
        return True
    else:
        logging.info(f"Modelo {filename} score de {erro}, abaixo do exigido")
        return False
        
if __name__ == '__main__':
    train()