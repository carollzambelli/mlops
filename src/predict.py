"""
Predict script following:
- core configuration
- pipeline data transformation
- trained ml model
"""

import pandas as pd
import pickle
import logging
from src.core import config, PACKAGE_ROOT, ASSETS_PATH
from src.utils import validate_inputs
from src.pipeline import df_model

logging.basicConfig(level=logging.INFO)

def make_prediction():

    logging.info("Iniciado processo de aplicação do modelo")
    data = pd.read_csv(f'{PACKAGE_ROOT}/{config.ml_config.predict_data_path}')
    load_model = pickle.load(
        open(f'{ASSETS_PATH}/{config.ml_config.trained_model_file}', 'rb'))
    validated_data, errors = validate_inputs(raw_data=data, step = "pred")
    predictions = None

    if not errors:
        logging.info("Dados validados")
        df = df_model(validated_data)
        predictions = load_model.predict(df[config.ml_config.features])
        logging.info(f"{len(predictions)} predições realizadas")

    pd.DataFrame(predictions).to_csv(
        f'{PACKAGE_ROOT}/{config.ml_config.result_data_path}', index=False)
    
    logging.info("Predições salvas")
    return True

                   
if __name__ == '__main__':
    make_prediction()