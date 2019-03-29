# Tensorflow-Lite-Example

In this example, we construct three different neural networks models to convert to Tensorflow Lite file(.tflite).
Next, we evaluate the speed and accuracy difference between original model and tflite model of these three different models.

# Result
After transforming to tflite file, the accuracy does not drop in all three models.
Although the speed accelarates on 3-layer and 9-layer model, the speed drops significantly on resnet50 model.

# Possible reason
Because the speed-up of integer arithmetic require special/optimized instructions/kernels, while such optimizations are not done on desktop CPU.
