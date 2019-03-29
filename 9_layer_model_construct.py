from __future__ import absolute_import, division, print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import tensorflow as tf
from tensorflow import keras
print(tf.__version__)
tf.enable_eager_execution()
import sys
import os
import numpy as np
import time
import pathlib

### Create Dataset 10000 train data; 10000 test data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Second 9 layers data
train_images_sec = train_images[:10000]
test_images_sec = test_images[:10000]
test_images_sec = np.float32(test_images_sec)
train_labels = train_labels[:10000].astype('int64')
test_labels = test_labels[:10000].astype('int64')
# Translation of data  
train_images_sec4D = train_images_sec.reshape(train_images_sec.shape[0], 28, 28, 1).astype('float32')  
test_images_sec4D = test_images_sec.reshape(test_images_sec.shape[0], 28, 28, 1).astype('float32')  

# Standardize feature data  
train_images_sec4D_norm = train_images_sec4D / 255  
test_images_sec4D_norm = test_images_sec4D /255  

ds_sec = tf.data.Dataset.from_tensor_slices((test_images_sec4D_norm, test_labels)).batch(1)

print(train_images_sec4D.shape)

# Returns a short sequential model
def create_model_sec():
  model = tf.keras.models.Sequential([
    keras.layers.Conv2D(filters=16, kernel_size=(5,5),padding='same', input_shape=(28,28,1), activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2,2)),
    keras.layers.Conv2D(filters=36, kernel_size=(5,5),padding='same', input_shape=(28,28,1), activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2,2)),
    keras.layers.Dropout(0.25),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation=tf.keras.activations.softmax)
  ])
  # compile
  opt = tf.train.AdamOptimizer()
  model.compile(optimizer=opt, 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
  return model

# Create a basic model instance
model_sec = create_model_sec()
model_sec.summary()
model_sec.fit(train_images_sec4D_norm, train_labels, epochs=5)
# eval test
start = time.time()
loss, acc = model_sec.evaluate(test_images_sec4D_norm, test_labels, batch_size=1)
time_pass = time.time() - start
print("time pass =", time_pass)
print("trained model, accuracy: {:5.2f}%".format(100*acc))
print('loss=',loss)

def eval_model_original(predictions, mnist_ds):
  
  total_seen = 0
  num_correct = 0

  for i, (img, label) in enumerate(mnist_ds):
    total_seen += 1
    pred = np.argmax(predictions[i])
    if pred == label.numpy():
      num_correct += 1

    if total_seen % 500 == 0:
        print("Accuracy after %i images: %f" %
              (total_seen, float(num_correct) / float(total_seen)))

  return float(num_correct) / float(total_seen)

# predict test data
start = time.time()
predictions = model_sec.predict(test_images_sec4D_norm,batch_size=1)
eval_model_original(predictions, ds_sec)
time_elapsed = time.time() - start
print("prediction time_elapsed=",time_elapsed)

# save
saved_models_root = "./saved_models/second"
print("saved_models_root =", saved_models_root)
tf.contrib.saved_model.save_keras_model(model_sec, saved_models_root)

saved_models_dir = str(sorted(pathlib.Path(saved_models_root).glob("*"))[-1])
print("saved_models_dir =", saved_models_dir)
