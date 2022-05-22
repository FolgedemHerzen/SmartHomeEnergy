#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import Sequential
from keras import regularizers
from keras.layers import Dropout
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.optimizer_v2.adam import Adam
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


#Create a one-dimensional array holding 25 pieces of data, 
#the first 24 for each hourly guide price and the last column for the classification label.
columns = list(map(str, range(24))) + ['label']
#Loading data in csv format
price_data = pd.read_csv('./TrainingData.txt', sep=',', names=columns)

#The partitioning of the training data into the training set, validation set and testing set
train = price_data.sample(frac=0.7, random_state=7)
train_label = train.pop('label')
val = price_data.drop(train.index)
validation = val.sample(frac=0.5, random_state=7)
validation_label = validation.pop('label')
test = val.drop(validation.index)
test_label = test.pop('label')


# Scaling the data to between 0 and 1 accelerates the gradient convergence.
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train)
scaled_validation = scaler.transform(validation)
scaled_test = scaler.transform(test)

#A fully connected network using relu as the activation function
#The output layer uses sigmoid. The network outputs a probability range between 0 and 1
#Use binary_crossentropy to calculate the cross-entropy loss between the true and predicted labels.

# MLP defination

model = Sequential()
model.add(Dense(64, activation='relu',kernel_regularizer = regularizers.l2(0.002),input_shape=(24,)))
model.add(Dropout(0.5))
model.add(Dense(32,activation='relu',kernel_regularizer = regularizers.l2(0.003)))
model.add(Dropout(0.5))
model.add(Dense(32,activation='relu',kernel_regularizer = regularizers.l2(0.003)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.summary()

#model training
#The loss function (Loss) is 'binary_crossentropy'.
#The 'adam' gradient descent search algorithm was used.
#250 (epochs=250) training times.
#Specify validation sets for cross-validation.
model.compile(
    optimizer='adam',loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
history = model.fit(scaled_train, train_label, batch_size = 512, epochs = 200, validation_data=(scaled_validation, validation_label))


model.summary()


"""The images are plotted against the training loss and validation loss during the training process 
to see if the model fits properly."""
history_dict = history.history
history_dict.keys()
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1,len(loss_values)+1)
plt.figure(dpi = 200, figsize = (15,2))
plt.plot(epochs,loss_values,'bo', markersize=1,label='Training loss')
plt.plot(epochs,val_loss_values,'b',markersize=0.5,label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
import numpy as np
test_pred = model.predict(scaled_test)
print("MSE: ", metrics.mean_squared_error(test_label, test_pred))
print("RMSE: ", np.sqrt(metrics.mean_squared_error(test_label, test_pred)))



"""Returns a list of binary classes (0 or 1) predicted by the model, based on the given price curve in "TestingData.txt"

The input pricing data must be of the shape (samples,24,1), where 'samples' denotes the number of samples in the input. The returned array is of the shape (samples, 1)."""

pred_data = pd.read_csv('./TestingData.txt', sep=',',names=list(map(str, range(24))))
scaled_data = scaler.transform(pred_data)
prediction = (model.predict(scaled_data) > 0.5).astype("int8").flatten()
print(prediction)
os.makedirs('./testing_results/') 
predicted_data = pd.concat([pred_data, pd.Series(prediction, name='label')], axis=1)
print(predicted_data)
predicted_data.to_csv(os.path.join('./testing_results', 'TestingResults.txt'),header=False, index=False)




