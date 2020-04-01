import matplotlib.pyplot as plt
seed_value = 0

from keras.models import load_model
import os
os.environ['PYTHONHASHSEED'] = str(seed_value)
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
import tensorflow as tf
tf.set_random_seed(seed_value)
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(),config = session_conf)
K.set_session(sess)

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import keras

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

class_number = 1012
#--------------------------------------------------------------------------------------------

# train data
with open('train_data.pickle', 'rb') as f:
    Data = pickle.load(f)
    Data = pd.DataFrame(Data)

numtrain = class_number * 6 * 12
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
#--------------------------------------------------------------------------------------------
# load model and test data

cnn_model= load_model('0.97acc.h5')
Test = pd.read_csv('test_data_30.csv')
X_test = scaler.transform(Test.iloc[:,1:])

#--------------------------------------------------------------------------------------------
# choice the one you want to print, here we pick zif-7 as an example

image_arr=np.reshape(X_test[12],(2251,1)) #zif-7 M-1
 
preds = cnn_model.predict(image_arr.reshape(1, 2251, 1))
label = np.argmax(preds[0])
ZIF7_output = cnn_model.output[:, label]

# last_conv_layer = cnn_model.get_layer(cnn_model.layers[14])

grads = K.gradients(ZIF7_output, cnn_model.layers[12].output)[0]

pooled_grads = K.mean(grads, axis = (0, 1, 2))

iterate = K.function([cnn_model.input], [pooled_grads, cnn_model.layers[14].output[0]])

pooled_grads_value, conv_layer_output_value = iterate([image_arr.reshape(1,2251, 1)])


heatmap = np.mean(conv_layer_output_value, axis=-1)

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)


new_heatmap_mat = np.zeros((1000, heatmap.shape[0]))
for i in range(1000):
	new_heatmap_mat[i] = heatmap


plt.matshow(new_heatmap_mat)
plt.show()


#--------------------------------------------------------------------------------------------

import cv2


heatmap = cv2.resize(new_heatmap_mat, (2251, 1000))

heatmap = np.uint8(255 * heatmap)  # change to RGB

heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)   # apply on orginal pic

superimposed_img = heatmap * 0.4 + image_arr    # 0.4 is parameter of heatmap


# set the right location

cv2.imwrite(r'C:\Users\cam.jpg', superimposed_img)   
