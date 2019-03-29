import tensorflow as tf
import numpy as np
tf.enable_eager_execution()
import pathlib
import time

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# First 3 layers data
train_labels = train_labels[:10000].astype('int64')
test_labels = test_labels[:10000].astype('int64')

train_images = train_images[:10000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:10000].reshape(-1, 28 * 28) / 255.0
test_images = np.float32(test_images)

ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(1)

# path
saved_models_root = "./saved_models/first"
print("saved_models_root =", saved_models_root)
saved_models_dir = str(sorted(pathlib.Path(saved_models_root).glob("*"))[-1])
print("saved_models_dir =", saved_models_dir)

# write tflite file
converter = tf.contrib.lite.TFLiteConverter.from_saved_model(saved_models_dir)
tflite_model = converter.convert()
tflite_models_dir = pathlib.Path('./mnist_tflite_models/')
tflite_models_dir.mkdir(exist_ok=True, parents=True)
tflite_model_file = tflite_models_dir/"mnist_model.tflite"
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
print(eval_model(interpreter, ds))
time_elapsed = time.time() - start
print("time_elapsed =", time_elapsed)


