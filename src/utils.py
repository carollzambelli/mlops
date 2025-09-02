"""
Transformes utils class to be used in pipeline and data schema validation
"""

import pandas as pd
import numpy as np
from pydantic import ValidationError
from src.core import config, MultipleDataSchema


def validate_inputs(raw_data: pd.DataFrame, step):
    """Validade columns and data type follow the model requirements """

    errors = None
    features = config.data_config.quanli_variables + config.data_config.quanti_variables
    
    if step == "train":
        data = raw_data[features]
        try:
            MultipleDataSchema(train=data.replace({np.nan: None}).to_dict(orient="records"))
        except ValidationError as error:
            errors = error.json()
    
    elif step == "pred":
        data = raw_data[features + [config.ml_config.target]]
        try:
            MultipleDataSchema(pred=data.replace({np.nan: None}).to_dict(orient="records"))
        except ValidationError as error:
            errors = error.json()

    return data, errors