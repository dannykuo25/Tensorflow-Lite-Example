import tensorflow as tf
import numpy as np
tf.enable_eager_execution()
import pathlib
import time

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


# path
saved_models_root = "./saved_models/second"
print("saved_models_root =", saved_models_root)
saved_models_dir = str(sorted(pathlib.Path(saved_models_root).glob("*"))[-1])
print("saved_models_dir =", saved_models_dir)

# write tflite file
converter = tf.contrib.lite.TFLiteConverter.from_saved_model(saved_models_dir)
tflite_model = converter.convert()
tflite_models_dir = pathlib.Path('./mnist_tflite_models/')
tflite_models_dir.mkdir(exist_ok=True, parents=True)
tflite_model_file = tflite_models_dir/"mnist_model_sec.tflite"
tflite_model_file.write_bytes(tflite_model)

print('write to .tflite file ok')

# interpret
interpreter = tf.contrib.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

def eval_model(interpreter, ds):
	total_seen = 0
	num_correct = 0

	for img, label in ds:
		total_seen += 1
		interpreter.set_tensor(input_index, img)
		interpreter.invoke()
		predictions = interpreter.get_tensor(output_index)
		predictions = np.argmax(predictions[0])
		if predictions == label.numpy():
			num_correct += 1

		if total_seen % 500 == 0:
			print("Accuracy after %i images: %f" % (total_seen, float(num_correct) / float(total_seen)))

	return float(num_correct) / float(total_seen)

# eval
start = time.time()
eval_model(interpreter, ds_sec)
time_elapsed = time.time() - start
print("time_elapsed =", time_elapsed)


