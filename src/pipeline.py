import pandas as pd
from sklearn.pipeline import Pipeline
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from src.core import config
from src import utils as pp
from sklearn.preprocessing import MinMaxScaler

price_pipe = Pipeline(
        [
        # == DATA TYPE ====
        (
            "categorical",
            pp.DataTypeTransformer(
                variables=config.data_config.categorical_variables,
                data_type="categorical"
            )
        ),
        (
            "numerical",
            pp.DataTypeTransformer(
                variables=config.data_config.numerical_variables,
                data_type="numerical"
            )
        ),
        # ===== DEAL WITH MISSINGS =====
        (
            "categorical_missing",
            CategoricalImputer(
                imputation_method="missing",
                variables=config.data_config.categorical_variables
            )
        ),
        (
            "numerical_missing",
            MeanMedianImputer(
                imputation_method="mean",
                variables=config.data_config.numerical_variables
            )
        ),
        # == TEMPORAL VARIABLES ====
        (
            "new_temporal_variable",
            pp.TemporalVariableTransformer(
                variables=config.data_config.temporal_vars
            )
        ),

         # == CATEGORICAL ENCODING
        (
            "rare_label_encoder",
            pp.RareCategoriesTransformer(
                variables=config.data_config.rare_encode,
                rare_perc = 0.01

            ),
        ),
    ]
)

def df_model(df_raw):

    df_scaler = pd.DataFrame(
        MinMaxScaler().fit_transform(df_raw[config.data_config.scale_vars]), 
        index=df_raw.index,
        columns=config.data_config.scale_vars
        )
    
    df = pd.concat([
        price_pipe.fit_transform(df_raw).drop(config.data_config.scale_vars, axis=1),
        df_scaler],axis=1
        )
    
    
    return df
