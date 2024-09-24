from classificationModel import model
import tensorflow as tf
import coremltools as ct

# Save the Keras model in SavedModel format
model.save('my_model.h5')  # Save in TensorFlow SavedModel format

# # Now use the SavedModel format for Core ML conversion
# coreml_model = ct.convert('my_model.h5', source='tensorflow')

# # Save the Core ML model
# coreml_model.save('FruitClassifier.mlmodel')


# Load the Keras model
model = tf.keras.models.load_model('my_model.h5')

# Get the concrete function from the Keras model
input_shape = (1, 224, 224, 3)  # Adjust to your model's input shape
run_model = tf.function(lambda x: model(x))
concrete_func = run_model.get_concrete_function(tf.TensorSpec(input_shape, model.inputs[0].dtype))

# Convert to Core ML using the concrete function
coreml_model = ct.convert(concrete_func, source='tensorflow')

# Save the Core ML model
coreml_model.save('FruitClassifier.mlmodel')

