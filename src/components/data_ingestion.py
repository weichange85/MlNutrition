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

from model_trainer import modelTrainerConfig
from model_trainer import ModelTrainer


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
                val_ds,
                img_height,
                img_width
            )
            

        except Exception as e:
            raise CustomException(e, sys)







if __name__ == "__main__":
    obj=DataIngestion()
    train_ds, val_ds, img_height, img_width = obj.initiate_data_ingestion()

    model_trainer = ModelTrainer(train_ds=train_ds, val_ds=val_ds)
    model = model_trainer.initiate_model_trainer(img_height=180, img_width=180)
    img_array = model_trainer.load_and_process(
        data_path="C:\Projects\MlNutrition\prediction_data\pears.jpg",
        img_height=180,
        img_width=180
    )
    prediction_class, score = model_trainer.make_prediction(model=model, img_array=img_array)


    print(f"This image most likely belongs to {prediction_class} with a {score:.2f} percent confidence.")