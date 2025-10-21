"""
Train script following:
- core configuration
- pipeline data transformation
"""

import pandas as pd
import logging
import warnings
from sklearn.model_selection import train_test_split 
from sklearn.metrics import root_mean_squared_error 
from src.core import config, PACKAGE_ROOT
from src.utils import validate_inputs, train_pipe
from datetime import datetime
warnings.filterwarnings('ignore')

#logging.basicConfig(filename=f'{PACKAGE_ROOT}/datasets/logs/train_log.txt',
#                    level=logging.INFO)

logging.basicConfig(level=logging.INFO)

def train():
    dt = datetime.now()
    logging.info(f"{dt} - Iniciando treinamento")
    raw_data = pd.read_table(f'{PACKAGE_ROOT}/{config.ml_config.train_data_path}',sep="\t",header=0)
    validated_data, errors = validate_inputs(raw_data)
    features = config.data_config.quali_variables + config.data_config.quanti_variables


    if  not errors:
        y = validated_data[config.ml_config.target]
        X = validated_data[features]

        X_treino, X_teste, y_treino, y_teste = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=123)
        
        treino_predict_pipe = train_pipe(X_treino, y_treino)
        y_hat = treino_predict_pipe.fit_transform(X_teste)
        errors = []
        for i in range(len(config.ml_config.models)):
            erro = root_mean_squared_error(
                y_teste, y_hat[len(y_teste)*(i):len(y_teste)*(i+1)])
            errors.append(erro)

        best_index = errors.index(min(errors))
        best_model = config.ml_config.models[best_index]

        if  errors[best_index] <= config.ml_config.erro :
            logging.info(f"{dt} - Modelo {best_model} - rmse:{erro}")
            return True
        else:
            logging.info(f"{dt} - Modelo com rmse maior do que o exigido")
            return False
    else:
        logging.info("Problema nos dados")
        return errors
        
if __name__ == '__main__':
    train()