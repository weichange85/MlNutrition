import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import sys

from tensorflow import keras
from keras import layers
from keras.models import Sequential
from src.exception import CustomException


def create_model(train_ds, img_height, img_width):
    try:
        num_classes = len(train_ds.class_names)

        model = Sequential([
            layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)
        ])

        return model
    except Exception as e:
        raise CustomException(e, sys)

def compile_model(model):

        model.compile(
                optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
            )
    
def train_model(model, train_ds, val_ds, epochs):
    return model.fit(train_ds, validation_data=val_ds, epochs=epochs)
