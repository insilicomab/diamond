from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

from src.middleware.logger import configure_logger
from src.models.base_model import AbstractBaseModel
from src.models.preprocess import DataPreprocessPipeline

logger = configure_logger(name=__name__)


@dataclass
class Evaluation:
    eval_df: pd.DataFrame
    mean_absolute_error: float
    mean_absolute_percentage_error: float
    mean_squared_error: float
    root_mean_squared_error: float


class Artifact(BaseModel):
    preprocess_file_path: Optional[str]
    model_file_path: Optional[str]


class Trainer(object):
    def __init__(self):
        pass

    def train(
        self,
        model: AbstractBaseModel,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_test: pd.DataFrame,
    ):
        model.train(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
        )

    def evaluate(
        self,
        model: AbstractBaseModel,
        x: pd.DataFrame,
        y: pd.DataFrame,
    ) -> Evaluation:
        predictions = model.predict(x=x)

        mse = mean_squared_error(
            y_true=y,
            y_pred=predictions,
            squared=True,
        )
        rmse = mean_squared_error(
            y_true=y,
            y_pred=predictions,
            squared=False,
        )
        mae = mean_absolute_error(
            y_true=y,
            y_pred=predictions,
        )
        mape = mean_absolute_percentage_error(
            y_true=y,
            y_pred=predictions,
        )
        evaluation = Evaluation(
            eval_df=x,
            mean_absolute_error=mae,
            mean_absolute_percentage_error=mape,
            mean_squared_error=mse,
            root_mean_squared_error=rmse,
        )
        logger.info(
            f"""
model: {model.name}
mae: {evaluation.mean_absolute_error}
mape: {evaluation.mean_absolute_percentage_error}
mse: {evaluation.mean_squared_error}
rmse: {evaluation.root_mean_squared_error}
            """
        )
        return evaluation

    def train_and_evaluate(
        self,
        model: AbstractBaseModel,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_test: pd.DataFrame,
        data_preprocess_pipeline: Optional[DataPreprocessPipeline] = None,
        preprocess_pipeline_file_path: Optional[str] = None,
        save_file_path: Optional[str] = None,
    ) -> Tuple[Evaluation, Artifact]:
        logger.info("start training and evaluation")
        self.train(
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
        )
        evaluation = self.evaluate(
            model=model,
            x=x_test,
            y=y_test,
        )

        artifact = Artifact()
        if (
            data_preprocess_pipeline is not None
            and preprocess_pipeline_file_path is not None
        ):
            artifact.preprocess_file_path = data_preprocess_pipeline.dump_pipeline(
                file_path=preprocess_pipeline_file_path
            )

        if save_file_path is not None:
            artifact.model_file_path = model.save(file_path=save_file_path)

        logger.info("done training and evaluation")
        return evaluation, artifact
