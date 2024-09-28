import pandas as pd
import os, sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.logger import logging
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
from dataclasses import dataclass


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
            pass
        except:
            pass







if __name__ == "__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()
