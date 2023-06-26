import os
from datetime import datetime

import hydra
import mlflow
from omegaconf import DictConfig

from src.jobs.retrieve import DataRetriever
from src.middleware.logger import configure_logger
from src.models.preprocess import DataPreprocessPipeline

logger = configure_logger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    logger.info("start ml...")
    logger.info(f"config: {cfg}")
    cwd = os.getcwd()

    logger.info(f"current working directory: {cwd}")
    logger.info(f"run_name: {cfg.run_name}")

    mlflow.set_experiment(cfg.run_name)
    with mlflow.start_run(run_name=cfg.run_name):
        data_retriever = DataRetriever()
        raw_df = data_retriever.retrieve_dataset(
            file_path=cfg.train_file_path,
        )
        data_preprocess_pipeline = DataPreprocessPipeline()
        data_preprocess_pipeline.preprocess(raw_df)


if __name__ == "__main__":
    main()
