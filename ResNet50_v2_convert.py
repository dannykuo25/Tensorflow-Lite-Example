import tensorflow as tf
import pathlib

'''
### Test tensorflow ###
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4

a = tf.constant(2)
b = tf.constant(3)

with tf.Session(config = config) as sess:
	print("a = 2, b = 3")
	print("Addition with constants: %i" % sess.run(a+b))
'''

filepath = "Resnet50_1.h5"

converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file(filepath)
tflite_model = converter.convert()
tflite_models_dir = pathlib.Path("./tflite_models_resnet50/")
tflite_models_dir.mkdir(exist_ok=True, parents=True)
tflite_model_file = tflite_models_dir/"model_resnet50_1.tflite"
tflite_model_file.write_bytes(tflite_model)

print("done")
