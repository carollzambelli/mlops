"""
Train script following:
- core configuration
- pipeline data transformation
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from src.core import config, PACKAGE_ROOT, ASSETS_PATH
from src.utils import validate_inputs
from src.pipeline import df_model


def train():

    data = pd.read_csv(f'{PACKAGE_ROOT}/{config.ml_config.train_data_path}')
    validated_data, errors = validate_inputs(raw_data=data, step = "train")

    if not errors:
        _df = df_model(validated_data)
        df = _df[config.ml_config.features + [config.ml_config.target]]
        X_train, X_test, y_train, y_test = train_test_split(
            df.drop([config.ml_config.target], axis=1), 
            df[config.ml_config.target], 
            test_size=0.20, 
            random_state=0
        )
        y_train = np.log(y_train)
        y_test = np.log(y_test)

        clf = RandomForestRegressor()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        if r2_score(y_test, y_pred) >= config.ml_config.r2_score_limit :
            filename = f'{ASSETS_PATH}/{config.ml_config.trained_model_file}'
            pickle.dump(clf, open(filename, 'wb'))
            return True
        else:
            return False
        
if __name__ == '__main__':
    train()