import os
from datetime import datetime

import hydra
import mlflow
from omegaconf import DictConfig

from src.jobs.predict import Predictor
from src.jobs.register import DataRegister
from src.jobs.retrieve import DataRetriever
from src.jobs.train import Trainer
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

        if cfg.jobs.train.run:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(f"outputs/{now}", exist_ok=True)
            preprocess_pipeline_file_path = os.path.join(
                cwd, f"outputs/{now}/pipeline_{model.name}_{now}"
            )
            trainer = Trainer()
            for i, dataset in enumerate(cross_validation_datasets):
                save_file_path = os.path.join(
                    cwd, f"outputs/{now}/{model.name}_{now}_{i}"
                )

                x_train, x_valid, y_train, y_valid = dataset
                evaluation, artifact = trainer.train_and_evaluate(
                    model=model,
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_valid,
                    y_test=y_valid,
                    data_preprocess_pipeline=data_preprocess_pipeline,
                    preprocess_pipeline_file_path=preprocess_pipeline_file_path,
                    save_file_path=save_file_path,
                )
                mlflow.log_metric("mean_absolute_error", evaluation.mean_absolute_error)
                mlflow.log_metric(
                    "mean_absolute_percentage_error",
                    evaluation.mean_absolute_percentage_error,
                )
                mlflow.log_metric("mean_squared_error", evaluation.mean_squared_error)
                mlflow.log_metric(
                    "root_mean_squared_error", evaluation.root_mean_squared_error
                )
                mlflow.log_artifact(artifact.preprocess_file_path, "preprocess")
                mlflow.log_artifact(artifact.model_file_path, "model")

                if cfg.jobs.predict.run:
                    predictor = Predictor()
                    predictions = predictor.predict(model=model, x=x_test, y=y_test)
                    logger.info(f"predictions: {predictions}")
                    print(predictions)

                    if cfg.jobs.predict.register:
                        data_register = DataRegister()
                        prediction_file_path = os.path.join(
                            cwd, f"outputs/{now}/prediction_{model.name}_{now}_{i}"
                        )
                        prediction_file_path = data_register.register(
                            predictions=predictions,
                            prediction_file_path=prediction_file_path,
                        )
                        mlflow.log_artifact(prediction_file_path, "prediction")

                mlflow.log_artifact(
                    os.path.join(cwd, "config/config.yaml"), "config.yaml"
                )

                mlflow.log_param("model", model.name)
                mlflow.log_params(model.params)


if __name__ == "__main__":
    main()
