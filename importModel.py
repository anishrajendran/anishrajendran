import tensorflow as tf

# Load MobileNetV2 pre-trained on ImageNet
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False,  # Exclude the final classification layer
                                               weights='imagenet')

# Freeze the base model layers so they are not trained
base_model.trainable = False
