import os
import sys
from dataclasses import dataclass

import polars as pl
from sklearn.model_selection import train_test_split

from end_to_end_ml_project_yt.exception import CustomException
from end_to_end_ml_project_yt.logger import logging


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "data.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self, data_path="notebook/data/stud.csv"):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pl.read_csv(data_path)
            logging.info("Read the dataset as dataframe")

            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )

            df.write_csv(self.ingestion_config.raw_data_path, include_header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.write_csv(
                self.ingestion_config.train_data_path, include_header=True
            )
            test_set.write_csv(
                self.ingestion_config.test_data_path, include_header=True
            )

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
