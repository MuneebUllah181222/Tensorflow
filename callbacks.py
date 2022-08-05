# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 01:04:07 2022

@author: Dell
"""

from tensorflow.keras.callbacks import Callback

class TrainingCallback(Callback):
    def on_train_begin(self, logs=None):
        print("Starting training...........")
    def on_epoch_begin(self, epoch, logs = None):
        print(f"staring epoch {epoch}")
    def on_batch_begin(self, batch, logs = None):
        print(f"Training: Starting batch {batch}")
    def on_batch_end(self, batch, logs = None):
        print(f"Training: Finished batch {batch} ")
    def on_epoch_end(self, epoch, logs=None):
        print(f"Finished epoch {epoch}")
    def on_train_end(self, logs = None):
        print("Finished training")

#create the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
from sklearn.datasets import load_diabetes

diabetes_dataseet = load_diabetes()
print(diabetes_dataseet["DESCR"])

#save the input data
print(diabetes_dataseet.keys())
data = diabetes_dataseet["data"]
targets = diabetes_dataseet["target"]

#scale the targets
targets = (targets-targets.mean(axis=0))/targets.std()

#spliting the data
from sklearn.model_selection import train_test_split
train_data, test_data, train_targets, tes_targets = train_test_split(data, targets, test_size = 0.1)

def get_regularize_model(wd, rate):
    model = Sequential([
        Dense(128, activation='relu',kernel_regularizer = regularizers.l2(wd),input_shape = (train_data.shape[1],)),
        Dropout(rate),
        Dense(128, kernel_regularizer = regularizers.l2(wd), activation='relu'),
        Dropout(rate),
        Dense(128, kernel_regularizer = regularizers.l2(wd), activation='relu'),
        Dropout(rate),
        Dense(128, kernel_regularizer = regularizers.l2(wd), activation='relu'),
        Dropout(rate),
        Dense(128, kernel_regularizer = regularizers.l2(wd),activation='relu'),
        Dropout(rate),
        Dense(128, kernel_regularizer = regularizers.l2(wd), activation='relu'),
        Dense(1)
        ])
    return model
model = get_regularize_model(1e-5, 0.3)
model.compile(optimizer="adam", loss="mae")

#train the model with callbacks
model.fit(train_data, train_targets,
          epochs=3,
          batch_size=128,
          verbose=False,
          callbacks=[TrainingCallback()])
#testing callbacks
class testing_callbacks(Callback):
    def on_test_begin(self, logs = None):
        print("Testing starting..........")
    def on_test_batch_begin(self, batch, logs = None):
        print(f"Testing: starting batch {batch}")
    def on_test_batch_end(self, batch, logs = None):
        print(f" Testing: Finished batch {batch}")
    def on_test_end(self, logs = None):
        print("Testing finished")

model.evaluate(test_data, tes_targets, verbose=False,
               callbacks = [testing_callbacks()])

#making prediction with callbacks
class predicting_callbacks(Callback):
    def on_predict_begin(self, logs = None):
        print("Prediction starting.........")
    def on_predict_batch_begin(self, batch, logs = None):
        print(f"Prediction: starting batch {batch}")
    def on_predict_batch_end(self, batch, logs = None):
        print(f"prediction: finished batch {batch}")
    def on_predict_end(self, logs= None):
        print("Prediction ends")

model.predict(test_data, verbose=False,
              callbacks = [predicting_callbacks()])

