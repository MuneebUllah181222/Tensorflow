# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 00:11:19 2022

@author: Dell
"""

#loading the adata
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

#train the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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

model = get_model()
model.summary()

#compile the model
model.compile(optimizer='adam',
              loss = "mse",
              metrics = ['mae'],
              )

history = model.fit(train_data, train_targets,
                    epochs=100,
                    validation_split=0.15,
                    verbose=False,
                    batch_size=64)
model.evaluate(test_data, tes_targets, verbose=False)

#plot the learning curves
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("epochs vs loss")
plt.ylabel("loss")
plt.xlabel("epochs")
plt.legend(["training","validation"], loc="upper right")
plt.show()




