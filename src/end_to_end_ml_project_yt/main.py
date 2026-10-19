import components.data_ingestion
import components.data_transformation

if __name__ == "__main__":
    obj = components.data_ingestion.DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = components.data_transformation.DataTransformation()
    data_transformation.initiate_data_transformation(
        train_data, test_data, target_col="math_score"
    )
