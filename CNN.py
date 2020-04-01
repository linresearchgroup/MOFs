import tensorflow as tf
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import keras
import random
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, CuDNNLSTM, Flatten
from keras.layers import AveragePooling1D, MaxPooling1D, Bidirectional, GlobalMaxPool1D, Concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D,concatenate
from keras.layers import SpatialDropout1D
from keras.callbacks import Callback
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import os
from math import floor
import warnings
from keras import backend as K
warnings.filterwarnings('ignore')

np.random.seed(42)
random.seed(12345)
#--------------------------------------------------------------------------------------------

# you need set your input classes number & output classes number here

input_class  = 1012
output_class = 1012

# If you want to train the model by yourself

# train data
with open('new_data_filter0.pickle', 'rb') as f:
    Data = pickle.load(f)
    Data = pd.DataFrame(Data)



numtrain = input_class * 6 * 12
numval = int(numtrain * 0.2)

# Train validation split
idx = random.sample(range(numtrain), numval) # 80% for training
all_index = list(np.setdiff1d(range(numtrain), idx))


# Getting X_train, X_val, y_train and scale them

X_train = Data.iloc[all_index,:].drop(2251, axis = 1)
X_val = Data.iloc[idx,:].drop(2251, axis = 1)
y_train = pd.get_dummies(Data.iloc[all_index,:][2251])

names = y_train.columns
mapping = {}
i = 0
for n in names:
    mapping[i] = n
    i+=1

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Construct CNN layers
# more para-LeNet-5
cnn_model = Sequential()
cnn_model.add(Conv1D(6, 5, strides = 1, activation = 'relu', input_shape = (2251, 1)))
cnn_model.add(MaxPooling1D(2, strides = 2))
cnn_model.add(Dropout(0.2))

cnn_model.add(Conv1D(16, 5, strides = 1, activation = 'relu'))
cnn_model.add(Conv1D(16, 5, strides = 1, activation = 'relu'))
cnn_model.add(MaxPooling1D(2, strides = 2))
cnn_model.add(Dropout(0.2))

cnn_model.add(Conv1D(32, 5, strides = 1, activation = 'relu'))
cnn_model.add(Conv1D(32, 5, strides = 1, activation = 'relu'))
cnn_model.add(MaxPooling1D(2, strides = 2))
cnn_model.add(Dropout(0.2))

cnn_model.add(Conv1D(64, 5, strides = 1, activation = 'relu'))
cnn_model.add(Conv1D(64, 5, strides = 1, activation = 'relu'))
cnn_model.add(MaxPooling1D(2, strides = 2))
cnn_model.add(Dropout(0.2))

cnn_model.add(Flatten())
cnn_model.add(Dense(120, activation = 'relu'))
cnn_model.add(Dense(84, activation = 'relu'))
cnn_model.add(Dense(output_class, activation = 'softmax', activity_regularizer = keras.regularizers.l2(0.1)))
cnn_model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


# Fit
numrest = numtrain-numval
cnn_model.fit(X_train.reshape(numrest, 2251, 1), y_train.values, batch_size = 128, epochs = 50, verbose = 1)

# Validate
pred = cnn_model.predict(X_val.reshape(numval, 2251, 1))
out = [np.argmax(p) for p in pred]
out = np.array(out)

# Mapping 
y_val = Data.iloc[idx,:][2251]
    
# Calculating accuracy
predicted = np.vectorize(mapping.get)(out)
print('Accuracy: '+ str(round(np.sum(predicted == y_val)/numval,2) * 100) + '%')

#--------------------------------------------------------------------------------------------

# Calculating accuracy for experimental data

Test = pd.read_csv('test_data_30.csv')

X_test = scaler.transform(Test.iloc[:,1:])
pred = cnn_model.predict(X_test.reshape(30, 2251, 1))
out = [np.argmax(p) for p in pred]
out = np.array(out)
predicted = np.vectorize(mapping.get)(out)

preds = []
for p in pred:
  P = []
  for i in np.argsort(p)[::-1][0:10]:
    P.append(mapping[i])
  preds.append(P)

for p in preds:
  print(p)