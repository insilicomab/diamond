import os
from typing import Dict, List, Optional, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor

from src.middleware.logger import configure_logger
from src.models.base_model import AbstractBaseModel

logger = configure_logger(__name__)

LGB_REGRESSION_DEFAULT_PARAMS = {
    "num_leaves": 122,
    "min_data_in_leaf": 8,
    "max_bin": 365,
    "bagging_fraction": 0.9213182882380164,
    "feature_fraction": 0.4980655277580941,
    "min_gain_to_split": 0.012265096895607893,
    "lambda_l1": 0.16090519318037727,
    "lambda_l2": 0.5600789957105483,
    "objective": "mae",
    "random_seed": 1234,
    "learning_rate": 0.02,
    "min_data_in_bin": 3,
    "bagging_freq": 1,
    "bagging_seed": 0,
    "verbose": -1,
}


class LightGBMRegression(AbstractBaseModel):
    def __init__(
        self,
        params: Dict = LGB_REGRESSION_DEFAULT_PARAMS,
        stopping_rounds: int = 100,
        eval_metrics: Union[str, List[str]] = "mse",
        verbose_eval: int = 500,
    ):
        self.name = "light_gbm_regression"
        self.params = params
        self.stopping_rounds = stopping_rounds
        self.eval_metrics = eval_metrics
        self.verbose_eval = verbose_eval

        self.model: LGBMRegressor = None
        self.reset_model(params=self.params)
        self.column_length: int = 0

    def reset_model(
        self,
        params: Optional[Dict] = None,
    ):
        if params is not None:
            self.params = params
        logger.info(f"params: {self.params}")
        self.model = LGBMRegressor(**self.params)
        logger.info(f"initialized model: {self.model}")

    def train(
        self,
        x_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.DataFrame],
        x_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    ):
        logger.info(f"start train for model: {self.model}")
        eval_set = [(x_train, y_train)]
        if x_test is not None and y_test is not None:
            eval_set.append((x_test, y_test))
        self.model.fit(
            X=x_train,
            y=y_train,
            eval_set=eval_set,
            callbacks=[
                lgb.early_stopping(self.stopping_rounds, verbose=True),
                lgb.log_evaluation(self.verbose_eval),
            ],
            eval_metric=self.eval_metrics,
        )

    def predict(
        self,
        x: Union[np.ndarray, pd.DataFrame],
    ) -> Union[np.ndarray, pd.DataFrame]:
        predictions = self.model.predict(x)
        return predictions

    def save_model_params(
        self,
        file_path: str,
    ) -> str:
        file, ext = os.path.splitext(file_path)
        if ext != ".yaml":
            file_path = f"{file}.yaml"
        logger.info(f"save model params: {file_path}")
        with open(file_path, "w") as f:
            yaml.dump(self.params, f)
        return file_path

    def save(
        self,
        file_path: str,
    ) -> str:
        file, ext = os.path.splitext(file_path)
        if ext != ".txt":
            file_path = f"{file}.txt"
        logger.info(f"save model: {file_path}")
        self.model.booster_.save_model(file_path)
        return file_path

    def load(
        self,
        file_path: str,
    ):
        logger.info(f"load model: {file_path}")
        self.model = lgb.Booster(model_file=file_path)
