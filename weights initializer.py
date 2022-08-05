# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 19:43:31 2022

@author: Dell
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
import tensorflow as tf

#weights initialization
model = Sequential([
    Conv1D(filters=16, kernel_size=3, input_shape = (128,64),
           kernel_initializer="random_uniform",
           bias_initializer="zeros",
           activation='relu'),
    MaxPooling1D(pool_size=4),
    Flatten(),
    Dense(64, kernel_initializer="he_uniform", bias_initializer="ones",
          activation="relu")
    ])
#another way
model.add(Dense(
    64,
    kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
    bias_initializer = tf.keras.initializers.Constant(value=0.5),
    activation = "relu"
    ))
model.add(Dense(
    64,
    kernel_initializer=tf.keras.initializers.Orthogonal(gain = 0.1, seed = None),
    bias_initializer=tf.keras.initializers.constant(value = 0.4)
    ))

model.summary()

import tensorflow.keras.backend as k
def my_init(shape, dtype = None):
    k.random_normal(shape, dtype=dtype)
model.add(Dense(
    64,
    kernel_initializer=my_init
    ))
