import tensorflow as tf
import pathlib
import time

import numpy as np
import keras
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras import optimizers

# from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(0)

from keras.datasets import cifar10

# set batch size and number of classes
input_shape = (32,32,3)
batch_size = 1
num_classes = 10
epochs = 10

# load cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# convert class vectors to one-hot binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# normalize data to [-1, 1]
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 127.5
x_test /= 127.5
x_train -= 1.
x_test -= 1.

x_train = x_train
y_train = y_train
x_test = x_test[:1000]
y_test = y_test[:1000]
print ("number of training examples = " + str(x_train.shape[0]))
print ("number of test examples = " + str(x_test.shape[0]))
print ("X_train shape: " + str(x_train.shape))
print ("Y_train shape: " + str(y_train.shape))
print ("X_test shape: " + str(x_test.shape))
print ("Y_test shape: " + str(y_test.shape))

filepath = "Resnet50_1.h5"
model = load_model(filepath) 
opt = optimizers.Adam()
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# # Evaluate the original model speed & accuracy

def eval_model_resnet50(predictions, y_test):
    
    label_final = np.argmax(predictions, axis=1)
    pred_final = np.argmax(y_test, axis=1)
    
    prediction_size = pred_final.size
    total_seen = 0
    num_correct = 0
    
    for i in range(0,prediction_size):
        total_seen += 1
        if pred_final[i] == label_final[i]:
            num_correct += 1
        if total_seen % 500 == 0:
            print("Accuracy after %i images: %f" %
                 (total_seen, float(num_correct) / float(total_seen)))
    return float(num_correct) / float(total_seen)

import time
# 1 model.predict test data...
print("model.predict test data...")
print("batch_size =", batch_size)
start = time.time()
predictions = model.predict(x_test, batch_size = batch_size)
eval_model_resnet50(predictions, y_test)
time_elapsed = time.time() - start
print("time_elapsed=",time_elapsed)
print("------------------")

# 2 model.evaluate test function
print("model.evaluate test data...")
print("batch_size =", batch_size)
start = time.time()
preds = model.evaluate(x_test, y_test, batch_size = batch_size)
time_elapsed = time.time() - start
print("time pass=", time_elapsed)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

# 3 model.evaluate train function
print("model.evaluate train data...")
start = time.time()
preds = model.evaluate(x_train, y_train, batch_size = batch_size)
time_elapsed = time.time() - start
print("time pass=", time_elapsed)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
