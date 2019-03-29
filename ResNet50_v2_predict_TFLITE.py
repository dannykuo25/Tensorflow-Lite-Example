import tensorflow as tf
import pathlib
import time

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4

tf.enable_eager_execution(config = config)

import keras
import numpy as np
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

ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

tflite_models_dir = pathlib.Path("./tflite_models_resnet50/")
tflite_model_file = tflite_models_dir/"model_resnet50_1.tflite"

interpreter = tf.contrib.lite.Interpreter(model_path=str(tflite_model_file))
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]
interpreter.resize_tensor_input(
    input_index,
    (batch_size, 32, 32, 3)
)
interpreter.allocate_tensors()

def eval_model(interpreter, mnist_ds):

  total_seen = 0
  num_correct = 0

  for img, label in mnist_ds:
    total_seen += int(img.shape[0])
    interpreter.set_tensor(input_index, img)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_index)
    predictions = np.argmax(predictions, axis = 1)
    num_correct += (predictions == np.argmax(label, axis = 1)).sum()

    if total_seen % 500 == 0:
        print("Accuracy after %i images: %f" %
              (total_seen, float(num_correct) / float(total_seen)))

  return float(num_correct) / float(total_seen)

print("start evaluate test data speed...")
start = time.time()
print(eval_model(interpreter, ds))
time_elapsed = time.time() - start
print("time_elapsed=",time_elapsed)


