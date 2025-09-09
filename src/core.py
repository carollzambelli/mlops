"""
teste Configuration of all relevant parameters to use in the project and data format validation
"""

from pathlib import Path
from typing import Dict, List
from pydantic import BaseModel
from strictyaml import load

PACKAGE_ROOT = Path().resolve()
ASSETS_PATH =  PACKAGE_ROOT / "assets"
CONFIG_FILE_PATH = ASSETS_PATH / "config.yml"

class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """
    target: str
    production_model_file: str
    trained_model_file: str
    train_data_path : str
    predict_data_path : str
    result_data_path : str
    models: List[str]
    erro : float

class DataConfig(BaseModel):
    """
    All configuration relevant to data
    sanitization and transformer classes
    """

    quali_variables: List[str]
    quanti_variables: List[str]

class Config(BaseModel):
    """Master config object."""

    data_config: DataConfig
    ml_config: ModelConfig

class TrainDataSchema(BaseModel):
    """
    Data Input schema
    """
    IDADE_CLIENTE: int
    RENDA_MENSAL_CLIENTE: float
    BEHAVIOUR_SCORE_CLIENTE: float
    QTD_TRANSACOES_3M: float
    QTD_ITENS_3M: float
    VALOR_GASTO_3M: float
    TICKET_MEDIO_3M: float
    FLAG_ELETRONICOS_3M: int
    SATISFACAO_ULTIMA_COMPRA: str

class MultipleDataSchema(BaseModel):
    inputs: List[TrainDataSchema]

def create_and_validate_config(cfg_path = CONFIG_FILE_PATH) -> Config:
    """Run validation on config values."""

    parsed_config = None
    try:
        with open(CONFIG_FILE_PATH, "r") as conf_file:
            parsed_config = load(conf_file.read())
    except:
        raise OSError(f"Did not find config file at path: {CONFIG_FILE_PATH}")

    
    _config = Config(
        data_config=DataConfig(**parsed_config.data),
        ml_config=ModelConfig(**parsed_config.data),
    )

    return _config


config = create_and_validate_config()


if __name__ == '__main__':
    print(PACKAGE_ROOT, ASSETS_PATH, CONFIG_FILE_PATH) 
    print(config)