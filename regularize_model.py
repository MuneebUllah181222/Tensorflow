# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 00:47:49 2022

@author: Dell
"""

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
model.compile(optimizer = 'adam',
              loss = 'mse',
              metrics = ['mae'])

history = model.fit(train_data, train_targets,
                    epochs=100,
                    validation_split=0.15,
                    batch_size=64,
                    verbose=False)
model.evaluate(test_data, tes_targets, verbose=2)

#plotting the results
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("epochs vs loss")
plt.ylabel("loss")
plt.xlabel("epochs")
plt.legend(["training","validation"], loc="upper right")
plt.show()

