import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.dataset.data_manager import load_df_from_csv
from src.dataset.schema import BASE_SCHEMA
from src.middleware.logger import configure_logger

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
