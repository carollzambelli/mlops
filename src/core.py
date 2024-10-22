"""
Configuration of all relevant parameters to use in the project and data format validation
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
    features: List[str]
    trained_model_file: str
    train_data_path : str
    result_data_path : str
    predict_data_path : str
    r2_score_limit : float

class DataConfig(BaseModel):
    """
    All configuration relevant to data
    sanitization and transformer classes
    """

    input_data_train: List[str]
    input_data_pred: List[str]
    categorical_variables: List[str]
    numerical_variables: List[str]
    map_variables : List[str]
    rare_encode : List[str]
    scale_vars : List[str]
    temporal_vars : Dict[str, str]
    zipcode_encoded : Dict[int, str]
    view_encoded: Dict[int, str]
    condition_encoded: Dict[int, str]
    grade_encoded: Dict[int, str]


class Config(BaseModel):
    """Master config object."""

    data_config: DataConfig
    ml_config: ModelConfig

class DataSchema(BaseModel):
    """
    Data Input schema
    """
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: int
    sqft_basement: int
    yr_built: int
    yr_renovated: int
    zipcode: int
    sqft_living15: int
    sqft_lot15: int


class MultipleDataSchema(BaseModel):
    inputs: List[DataSchema]

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