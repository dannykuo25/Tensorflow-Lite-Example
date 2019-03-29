from __future__ import absolute_import, division, print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import tensorflow as tf
from tensorflow import keras

tf.__version__
tf.enable_eager_execution()

import sys
import os
import numpy as np
import time
import pathlib

### Create Dataset 10000 train data; 10000 test data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# First 3 layers data
train_labels = train_labels[:10000].astype('int64')
test_labels = test_labels[:10000].astype('int64')

train_images = train_images[:10000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:10000].reshape(-1, 28 * 28) / 255.0

ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(1)

# Returns a short sequential model
def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation=tf.keras.activations.relu, input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation=tf.keras.activations.softmax)
  ])
  opt = tf.train.AdamOptimizer()
  model.compile(optimizer=opt, 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])    
  return model

# Create a basic model instance
model = create_model()
model.summary()
model.fit(train_images, train_labels, epochs=5)
# eval test
start = time.time()
loss, acc = model.evaluate(test_images, test_labels)
end = time.time()
print("time pass =", end-start)
print("eval model, accuracy: {:5.2f}%".format(100*acc))
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
predictions = model.predict(test_images,batch_size=1)
eval_model_original(predictions, ds)
time_elapsed = time.time() - start
print("prediction time_elapsed=",time_elapsed)

# save
saved_models_root = "./saved_models/first"
print("saved_models_root =", saved_models_root)
tf.contrib.saved_model.save_keras_model(model, saved_models_root)

saved_models_dir = str(sorted(pathlib.Path(saved_models_root).glob("*"))[-1])
print("saved_models_dir =", saved_models_dir)
