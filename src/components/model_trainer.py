import os
import sys
from dataclasses import dataclass

import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras import layers
from keras.models import Sequential

from src.exception import CustomException
from src.logger import logging
from src.utils import create_model, compile_model, train_model


@dataclass
class modelTrainerConfig:
    trainer_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self, train_ds, val_ds):
        self.model_trainer_config = modelTrainerConfig()
        self.train_ds = train_ds
        self.val_ds = val_ds
    
    def initiate_model_trainer(self, img_height, img_width):
        
        model = create_model(train_ds=self.train_ds, img_height=img_height, img_width=img_width)

        compile_model(model=model)

        train_model(model=model, train_ds=self.train_ds, val_ds=self.val_ds, epochs=2)

        return model

    def load_and_process(self, data_path, img_height, img_width):
        img = tf.keras.utils.load_img(
            path=data_path, 
            target_size=(img_height,img_width)
        )

        img_array = tf.keras.utils.img_to_array(img=img)
        img_array = tf.expand_dims(img_array, 0)

        return img_array
    
    def make_prediction(self, model, img_array):
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        class_names = self.train_ds.class_names
        prediction_class = class_names[np.argmax(score)]
        
        return prediction_class, score

