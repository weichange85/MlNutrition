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

@dataclass
class data_standardization:
    def __init__(self):
        self.original_range_max: int = 255 #need refactor
        self.rescale_range_max: float = 1.

    def initialize_data_standardization(self, train_ds, test_ds):
        normalization_layer = tf.keras.layers.Rescaling(
            self.rescale_range_max/self.original_range_max
            )
        
        normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

        ##unfinished