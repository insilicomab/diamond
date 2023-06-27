import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from src.dataset.data_manager import load_df_from_csv
from src.dataset.schema import BASE_SCHEMA, RAW_PREDICTION_SCHEMA, X_SCHEMA, Y_SCHEMA
from src.middleware.logger import configure_logger
from src.models.preprocess import DataPreprocessPipeline

logger = configure_logger(__name__)


class DataRetriever:
    def __init__(self):
        pass

    def retrieve_dataset(self, file_path: str) -> pd.DataFrame:
        logger.info("start retrieve data")
        raw_df = load_df_from_csv(file_path)
        raw_df = BASE_SCHEMA.validate(raw_df)
        logger.info(
            f"""
Loaded dataset
raw_df columns: {raw_df.columns}
raw_df shape: {raw_df.shape}
    """
        )
        return raw_df

    def train_test_split(
        self,
        raw_df: pd.DataFrame,
        data_preprocess_pipeline: DataPreprocessPipeline,
    ) -> tuple[
        list[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series], pd.DataFrame, pd.Series
    ]:
        _raw_df = data_preprocess_pipeline.preprocess(x=raw_df)
        X = _raw_df.drop(["price"], axis=1).reset_index(drop=True)
        y = _raw_df["price"].reset_index(drop=True)

        X = X_SCHEMA.validate(X)
        y = Y_SCHEMA.validate(y)

        x_train_val, x_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True, random_state=0
        )
        logger.info(
            f"""
weekly train df columns: {x_train_val.columns}
weekly train df shape: {x_train_val.shape}
weekly test df columns: {x_test.columns}
weekly test df shape: {x_test.shape}
    """
        )

        x_train_val = data_preprocess_pipeline.fit_transform(x=x_train_val)
        x_test = data_preprocess_pipeline.transform(x=x_test)
        logger.info(
            f"""
preprocessed train df columns: {x_train_val.columns}
preprocessed train df shape: {x_train_val.shape}
preprocessed test df columns: {x_test.columns}
preprocessed test df shape: {x_test.shape}
    """
        )

        cross_validation_datasets = [
            (
                x_train_val.iloc[train_index],
                x_train_val.iloc[val_index],
                y_train_val.iloc[train_index],
                y_train_val.iloc[val_index],
            )
            for train_index, val_index in KFold(
                n_splits=5, shuffle=True, random_state=0
            ).split(x_train_val)
        ]
        logger.info("done split data")
        return (cross_validation_datasets, x_test, y_test)

    def retrieve_prediction_data(self, file_path: str) -> pd.DataFrame:
        logger.info("start retrieve prediction data")
        data_to_be_predicted_df = load_df_from_csv(file_path)
        data_to_be_predicted_df = RAW_PREDICTION_SCHEMA.validate(
            data_to_be_predicted_df
        )
        logger.info(
            f"""
Loaded dataset
raw_df columns: {data_to_be_predicted_df.columns}
raw_df shape: {data_to_be_predicted_df.shape}
    """
        )
        return data_to_be_predicted_df
