import os
import sys
from dataclasses import dataclass

import numpy as np
import polars as pl
import polars.selectors as cs
from pydantic import InstanceOf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from end_to_end_ml_project_yt.exception import CustomException
from end_to_end_ml_project_yt.logger import logging
from end_to_end_ml_project_yt.utils import save_object

PolarsDF = InstanceOf[pl.DataFrame]


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(
        self, numeric_features: list, categorical_features: list
    ):
        """
        This function is responsible for data transformation
        """
        try:
            num_pipeline = Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("standardise", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                [
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("encode", OneHotEncoder()),
                ]
            )

            logging.info(f"Categorical features: {categorical_features}")
            logging.info(f"Numerical features: {numeric_features}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numeric_features),
                    ("cat_pipeline", cat_pipeline, categorical_features),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path, target_col):
        logging.info("Start data transformaiton")

        try:
            train_df = pl.read_csv(train_path)
            test_df = pl.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            numeric_features = train_df.select(cs.numeric().exclude(target_col)).columns
            categorical_features = train_df.select(
                cs.string().exclude(target_col)
            ).columns

            preprocessing_obj = self.get_data_transformer_object(
                numeric_features, categorical_features
            )

            input_feature_train_df = train_df.drop(target_col)
            target_feature_train_df = train_df.get_column(target_col)

            input_feature_test_df = test_df.drop(target_col)
            target_feature_test_df = test_df.get_column(target_col)

            logging.info("Applying preprocessing object on train and test dataframe")

            input_feature_train_df = input_feature_train_df.to_pandas()
            input_feature_test_df = input_feature_test_df.to_pandas()

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
