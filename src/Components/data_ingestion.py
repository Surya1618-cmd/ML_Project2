import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass # Added this import

from src.exception import CustomException
from src.logger import get_logger

logging = get_logger(__name__)

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("Raw_data", "health_dataset_with_issues.csv") # Assuming this is your raw data

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion process.")
        try:
            df = pd.read_csv(self.ingestion_config.raw_data_path)
            logging.info(f"Raw data loaded from {self.ingestion_config.raw_data_path} with shape {df.shape}")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Split data into training and testing sets
            # For ingestion, we split the full raw data.
            # No target/feature separation here yet.
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info(f"Data ingestion complete. Train data saved to {self.ingestion_config.train_data_path}")
            logging.info(f"Test data saved to {self.ingestion_config.test_data_path}")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.error(f"Error in initiate_data_ingestion function: {e}", exc_info=True)
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()
    logging.info(f"Data ingestion completed. Train path: {train_path}, Test path: {test_path}")
