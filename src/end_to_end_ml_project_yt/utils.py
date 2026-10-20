import os
import sys

import dill
import mlflow
import mlflow.sklearn
from sklearn.metrics import r2_score

from end_to_end_ml_project_yt.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


mlflow.set_tracking_uri("http://localhost:5000")


def evaluate_models(
    X_train,
    y_train,
    X_test,
    y_test,
    models,
    experiment_name="model-evaluation",
    log_models=True,
):
    try:
        mlflow.set_experiment(experiment_name)

        mlflow.sklearn.autolog(log_models=log_models, silent=True)

        report = {}

        for name, model in models.items():
            with mlflow.start_run(run_name=name):
                mlflow.log_param("model_name", name)

                model.fit(X_train, y_train)

                y_test_pred = model.predict(X_test)

                test_model_score = r2_score(y_test, y_test_pred)

                mlflow.log_metric("test_r2", test_model_score)
                mlflow.set_tag("stage", "evaluation")

                report[name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
