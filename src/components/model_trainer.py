import os
import sys
from dataclasses import dataclass

import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds

from src.exception import CustomException
from src.logger import logging
# from src.utils import save_object, evaluate_models


@dataclass
class modelTrainerConfig:
    trainer_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = modelTrainerConfig()
    
    def initiate_model_trainer(self, train_data, test_data):
        pass