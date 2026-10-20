import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from end_to_end_ml_project_yt.exception import CustomException
from end_to_end_ml_project_yt.logger import logging
from end_to_end_ml_project_yt.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Split training and test input data")

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Cat Boost Regressor": CatBoostRegressor(verbose=False),
                "Ada Boost Regressor": AdaBoostRegressor(),
                "Gradient Boost Regressor": GradientBoostingRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "XG Boost Regressor": XGBRegressor(),
            }

            param_grid = {
                "Linear Regression": {},
                "Cat Boost Regressor": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100],
                },
                "Ada Boost Regressor": {
                    "learning_rate": [0.001, 0.01, 0.1, 0.5],
                    "n_estimators": [50, 100, 256],
                },
                "Gradient Boost Regressor": {
                    "learning_rate": [0.001, 0.01, 0.1],
                    "subsample": [0.7, 0.8, 0.9],
                    "n_estimators": [50, 100, 256],
                    "max_depth": [3, 5, 7],
                },
                "Random Forest Regressor": {
                    "n_estimators": [100, 256],
                    "max_depth": [10, 15, 20, None],
                    "min_samples_leaf": [1, 2, 4],
                },
                "K-Neighbors Regressor": {"n_neighbors": [5, 7, 9, 11]},
                "Decision Tree Regressor": {
                    "criterion": ["squared_error", "friedman_mse"],
                    "max_depth": [3, 5, 7, 10, None],
                },
                "XG Boost Regressor": {
                    "learning_rate": [0.001, 0.01, 0.1],
                    "n_estimators": [50, 100, 256],
                    "max_depth": [3, 5, 7, 10],
                },
            }

            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=param_grid,
            )

            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info("Best model found on training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)
