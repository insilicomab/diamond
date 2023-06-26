from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.dataset.schema import BASE_SCHEMA
from src.middleware.logger import configure_logger

logger = configure_logger(__name__)

CATEGORICAL_FEATURES = ["cut", "color", "clarity"]


# 外れ値データの除外
def remove_outliter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(df[(df["x"] == 0) | (df["y"] == 0) | (df["z"] == 0)].index, axis=0)
    df = df.drop(df[(df["x"] >= 10) | (df["y"] >= 10) | (df["z"] >= 10)].index, axis=0)
    df.reset_index(inplace=True, drop=True)
    return df


# 密度（重さ/体積）
def calc_density(df: pd.DataFrame) -> pd.DataFrame:
    df["density"] = df["carat"] / (df["x"] * df["y"] * df["z"])
    return df


# 差分
def calc_diff(df: pd.DataFrame) -> pd.DataFrame:
    df["x-y"] = (df["x"] - df["y"]).abs()
    df["y-z"] = (df["y"] - df["z"]).abs()
    df["z-x"] = (df["x"] - df["y"]).abs()
    return df


# 比率
def calc_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df["x/y"] = df["x"] / df["y"]
    df["y/z"] = df["y"] / df["z"]
    df["z/x"] = df["z"] / df["x"]
    return df


# 中央値との差分
def calc_diff_from_median(df: pd.DataFrame) -> pd.DataFrame:
    df["x-median_x"] = (df["x"] - df["x"].median()).abs()
    df["y-median_y"] = (df["y"] - df["y"].median()).abs()
    df["z-median_z"] = (df["z"] - df["z"].median()).abs()
    return df


# カテゴリ変数cutごとにcarat中央値を集計
def agg_median_carat_by_cut(df: pd.DataFrame) -> pd.DataFrame:
    carat_by_cut = df.groupby("cut")["carat"].agg("median").reset_index()
    df = pd.merge(df, carat_by_cut, on="cut", how="left")
    return df


# caratとcarat中央値の差分
def diff_carat_median_carat_by_cut(df: pd.DataFrame) -> pd.DataFrame:
    df["carat-median_carat_by_cut"] = df["carat"] - df["median_carat_by_cut"]
    return df


# caratとcarat中央値の比率
def ratio_carat_per_median_carat_by_cut(df: pd.DataFrame) -> pd.DataFrame:
    df["carat/median_carat_by_cut"] = df["carat"] / df["median_carat_by_cut"]
    return df


# カテゴリ変数colorごとにcarat中央値を集計
def agg_median_carat_by_color(df: pd.DataFrame) -> pd.DataFrame:
    carat_by_color = df.groupby("color")["carat"].agg("median").reset_index()
    df = pd.merge(df, carat_by_color, on="color", how="left")
    return df


# caratとcarat中央値の差分
def diff_carat_median_carat_by_color(df: pd.DataFrame) -> pd.DataFrame:
    df["carat-median_carat_by_color"] = df["carat"] - df["median_carat_by_color"]
    return df


# caratとcarat中央値の比率
def ration_carat_median_carat_by_color(df: pd.DataFrame) -> pd.DataFrame:
    df["carat/median_carat_by_color"] = df["carat"] / df["median_carat_by_color"]
    return df


# カテゴリ変数clarityごとにcarat中央値を集計
def agg_carat_by_clarity(df: pd.DataFrame) -> pd.DataFrame:
    carat_by_clarity = df.groupby("clarity")["carat"].agg("median").reset_index()
    df = pd.merge(df, carat_by_clarity, on="clarity", how="left")
    return df


# caratとcarat中央値の差分
def diff_carat_median_carat_by_clarity(df: pd.DataFrame) -> pd.DataFrame:
    df["carat-median_carat_by_clarity"] = df["carat"] - df["median_carat_by_clarity"]
    return df


# caratとcarat中央値の比率
def ratio_carat_per_median_carat_by_clarity(df: pd.DataFrame) -> pd.DataFrame:
    df["carat/median_carat_by_clarity"] = df["carat"] / df["median_carat_by_clarity"]
    return df


# カテゴリ変数cut×colorで集計した出現割合の特徴量
def ratio_cut_times_color(df: pd.DataFrame) -> pd.DataFrame:
    # クロス集計表の出現割合
    cross = pd.crosstab(df["cut"], df["color"], normalize="index")
    cross = cross.reset_index()
    # クロス集計表のテーブルへの変換
    tbl = pd.melt(cross, id_vars="cut", value_name="rate_cut*color")
    # 出現割合の特徴量追加
    df = pd.merge(df, tbl, on=["cut", "color"], how="left")
    return df


# カテゴリ変数color×clarityで集計した出現割合の特徴量
def ratio_rate_color_times_clarity(df: pd.DataFrame) -> pd.DataFrame:
    # クロス集計表の出現割合
    cross = pd.crosstab(df["color"], df["clarity"], normalize="index")
    cross = cross.reset_index()
    # クロス集計表のテーブルへの変換
    tbl = pd.melt(cross, id_vars="color", value_name="rate_color*clarity")
    # 出現割合の特徴量追加
    df = pd.merge(df, tbl, on=["color", "clarity"], how="left")
    return df


# カテゴリ変数clarity×cutで集計した出現割合の特徴量
def ratio_clarity_times_cut(df: pd.DataFrame) -> pd.DataFrame:
    # クロス集計表の出現割合
    cross = pd.crosstab(df["clarity"], df["cut"], normalize="index")
    cross = cross.reset_index()
    # クロス集計表のテーブルへの変換
    tbl = pd.melt(cross, id_vars="clarity", value_name="rate_clarity*cut")
    # 出現割合の特徴量追加
    df = pd.merge(df, tbl, on=["clarity", "cut"], how="left")
    return df


class BasePreprocessPipeline(ABC, BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    @abstractmethod
    def fit(
        self,
        x: pd.DataFrame,
        y=None,
    ):
        raise NotImplementedError

    @abstractmethod
    def transform(
        self,
        x: pd.DataFrame,
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def fit_transform(
        self,
        x: pd.DataFrame,
        y=None,
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def dump_pipeline(
        self,
        file_path: str,
    ):
        raise NotImplementedError

    @abstractmethod
    def load_pipeline(
        self,
        file_path: str,
    ):
        raise NotImplementedError


class DataPreprocessPipeline:
    def __init__(self) -> None:
        self.pipeline: Union[Pipeline, ColumnTransformer] = None
        self.define_pipeline()

    def define_pipeline(self):
        categorical_pipeline = Pipeline(
            [
                (
                    "simple_imputer",
                    SimpleImputer(
                        missing_values=np.nan, strategy="constant", fill_value=None
                    ),
                ),
                ("ordinal_encoder", OrdinalEncoder(encoded_missing_value=-1)),
            ]
        )
        self.pipeline = ColumnTransformer(
            [
                ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ],
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")
        logger.info(f"pipeline: {self.pipeline}")

    def preprocess(
        self,
        x: pd.DataFrame,
        y=None,
    ) -> pd.DataFrame:
        x = BASE_SCHEMA.validate(x)
        x = remove_outliter(x)
        x = calc_density(x)
        print(x.head())
        print(x.shape)
        return x
