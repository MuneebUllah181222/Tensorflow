# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 01:31:57 2022

@author: Dell
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers 
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
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

def get_model():
    model = Sequential([
        Dense(128, activation='relu',input_shape = (train_data.shape[1],)),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(1)
        ])
    return model
unregularize_model = get_model()
unregularize_model.compile(optimizer = "adam", loss = "mae")
unreg_history = unregularize_model.fit(test_data, train_targets,
                            epochs=100,
                            validation_split=0.15,
                            batch_size=64,
                            verbose=False,
                            callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)]
                            )

unregularize_model.evaluate(test_data, tes_targets,
                            verbose=2)

#regularize model
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
regu_model = get_regularize_model(1e-5, 0.2)

regu_model.summary()
regu_model.compile(optimizer = "adam",
                   loss = "mse")
regu_history = regu_model.fit(train_data, train_targets,
                              epochs=100,
                              batch_size=64,
                              verbose=False,
                              validation_split=0.15,
                              callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)])


regu_model.evaluate(test_data, tes_targets, verbose=2)

#ploting the two models
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12,5))

fig.add_subplot(121)

plt.plot(unreg_history.history['loss'])
plt.plot(unreg_history.history['val_loss'])
plt.title("Unregularized:loss vs epochs")
plt.ylabel("loss")
plt.xlabel("epochs")
plt.legend(["Training","Validation"], loc = "upper right")

fig.add_subplot(122)

plt.plot(regu_history.history['loss'])
plt.plot(regu_history.history['val_loss'])
plt.title("Regularized:loss vs epochs")
plt.ylabel("loss")
plt.xlabel("epochs")
plt.legend(["Training", "validation"], loc = "upper right")

plt.show()

