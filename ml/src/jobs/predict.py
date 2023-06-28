import numpy as np
import pandas as pd

from src.dataset.schema import PREDICTION_SCHEMA
from src.middleware.logger import configure_logger
from src.models.base_model import AbstractBaseModel

logger = configure_logger(name=__name__)

PREDICTION_SCHEMA


class Predictor:
    def __init__(self):
        pass

    def postprocess(self, true: pd.Series, predictions: np.ndarray):
        df = pd.DataFrame({"true": true, "predictions": predictions})
        return df

    def predict(
        self, model: AbstractBaseModel, x: pd.DataFrame, y: pd.Series
    ) -> pd.DataFrame:
        predictions = model.predict(x=x)
        predictions = self.postprocess(true=y, predictions=predictions)
        predictions = PREDICTION_SCHEMA.validate(predictions)
        return predictions
