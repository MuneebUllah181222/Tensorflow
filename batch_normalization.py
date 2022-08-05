# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 20:35:37 2022

@author: Dell
"""
#batch normalization
import tensorflow as tf
tf.__version__
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout
from tensorflow.keras.models import Sequential

#load the data
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

diabetes_dataset = load_diabetes()
data = diabetes_dataset['data']
targets = diabetes_dataset['target']

targets = (targets-targets.mean(axis = 0))/(targets.std())

train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size = 0.1)

#defining model
model = Sequential()
model.add(Dense(
    64, input_shape = [train_data.shape[1],], activation = "relu",
))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(256, activation="relu", name="last"))


model.summary()

model.add(BatchNormalization(
    momentum=0.95,
    epsilon=0.005,
    axis=-1,
    beta_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
    gamma_initializer=tf.keras.initializers.Constant(value=0.9)
    ))

model.add(Dense(1))
model.summary()

model.compile(
    optimizer = "adam",
    loss = "mse",
    metrics = ["mae"]
    )

history = model.fit(
    test_data, train_targets,
    epochs=100,
    batch_size=64,
    verbose=False,
    validation_split=0.15
    )

#plotting the curves
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

frame = pd.DataFrame(history.history)
epochs = np.arange(len(frame))

fig = plt.figure(figsize=(12,4))

# Loss plot
ax = fig.add_subplot(121)
ax.plot(epochs, frame['loss'], label="Train")
ax.plot(epochs, frame['val_loss'], label="Validation")
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.set_title("Loss vs Epochs")
ax.legend()

# Accuracy plot
ax = fig.add_subplot(122)
ax.plot(epochs, frame['mae'], label="Train")
ax.plot(epochs, frame['val_mae'], label="Validation")
ax.set_xlabel("Epochs")
ax.set_ylabel("Mean Absolute Error")
ax.set_title("Mean Absolute Error vs Epochs")
ax.legend()

plt.show()
