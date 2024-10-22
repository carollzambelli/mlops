"""
Transformes utils class to be used in pipeline and data schema validation
"""

from typing import List, Dict
import pandas as pd
import numpy as np
from pydantic import ValidationError
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
from src.core import config, MultipleDataSchema


class DataTypeTransformer(BaseEstimator, TransformerMixin):
    """Data type transformer."""
    def __init__(self, variables: List[str], data_type: str):

        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables
        self.data_type = data_type

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need this step to fit the sklearn pipeline
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        # so that we do not over-write the original dataframe
        X = X.copy()

        for var in self.variables:

            if self.data_type == "categorical":
                X[var] = X[var].astype(str)
            elif self.data_type == "numerical":
                X[var] = X[var].astype(int)
            elif self.data_type == "date":
                X[var] = pd.to_datetime(X[var])
            
        return X

class TemporalVariableTransformer(BaseEstimator, TransformerMixin):
    """Temporal elapsed time transformer."""

    def __init__(self, variables: Dict[str, str]):

        if not isinstance(variables, dict):
            raise ValueError("variables should be a dict")

        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need this step to fit the sklearn pipeline
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        # so that we do not over-write the original dataframe
        X = X.copy()

        for key in self.variables.keys():
            X[key] = np.where(
                X[self.variables[key]] > 0,
                pd.to_datetime("now").year - X[self.variables[key]],
                0)
            
        return X
    
class MapperTransformer(BaseEstimator, TransformerMixin):
    """Column namme variable mapper."""

    def __init__(self, variables: str):

        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(config.ml_config.features)

        return X
    
class RareCategoriesTransformer(BaseEstimator, TransformerMixin):
    """Rare categorical variable mapper."""

    def __init__(self, variables: str, rare_perc : float):

        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables
        self.rare_perc = rare_perc

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for var in self.variables:

            tmp = X.groupby(var)[var].count() / len(X)
                
            frequent_ls = tmp[tmp >  self.rare_perc].index

            # substitui as categorias de baixa frequencia pela categoria "00000"
            X[var] = np.where(X[var].isin(frequent_ls), X[var], '00000')

            # Aplicar o encoder para a coluna
            encoder = OrdinalEncoder(categories=[X[var].unique()])

            X[f'{var}_encoded'] = encoder.fit_transform(X[[var]])

        return X


def validate_inputs(raw_data: pd.DataFrame, step):
    """Validade columns and data type follow the model requirements """

    errors = None
    if step == "train":
        data = raw_data[config.data_config.input_data_train]
    elif step == "pred":
        data = raw_data[config.data_config.input_data_pred]

    try:
        MultipleDataSchema(
            inputs=data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return data, errors