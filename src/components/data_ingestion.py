import pandas as pd
import os, sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.logger import logging
from src.exception import CustomException
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
from dataclasses import dataclass
import pathlib


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")

class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Enter Data Ingestion Method")
        try:
            data_dir = pathlib.Path("notebook\data")
            logging.info("Finished loading dataset")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            logging.info("Train test split initiated")

            batch_size = 32
            img_height = 180
            img_width = 180

            train_ds = tf.keras.utils.image_dataset_from_directory(
                data_dir,
                validation_split=0.2,
                subset="training",
                seed=123,
                image_size=(img_height, img_width),
                batch_size=batch_size)

            val_ds = tf.keras.utils.image_dataset_from_directory(
                data_dir,
                validation_split=0.2,
                subset="validation",
                seed=123,
                image_size=(img_height, img_width),
                batch_size=batch_size)
            
            logging.info("data ingestion complete, data split into train/validation sets")

            return(
                train_ds,
                val_ds
            )
            

        except Exception as e:
            raise CustomException(e, sys)







if __name__ == "__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()
