import os
from datetime import datetime

import hydra
import mlflow
from omegaconf import DictConfig

from src.jobs.retrieve import DataRetriever
from src.middleware.logger import configure_logger
from src.models.models import MODELS
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
            file_path=cfg.jobs.data.path,
        )
        data_preprocess_pipeline = DataPreprocessPipeline()
        cross_validation_datasets, x_test, y_test = data_retriever.train_test_split(
            raw_df=raw_df,
            data_preprocess_pipeline=data_preprocess_pipeline,
        )
        _model = MODELS.get_model(name=cfg.jobs.model.name)
        model = _model.model(
            stopping_rounds=cfg.jobs.model.params.stopping_rounds,
            eval_metrics=cfg.jobs.model.params.eval_metrics,
            verbose_eval=cfg.jobs.model.params.verbose_eval,
        )

        if "params" in cfg.jobs.model.keys():
            model.reset_model(params=cfg.jobs.model.params)


if __name__ == "__main__":
    main()
