from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create ImageDataGenerator objects for loading images
datagen = ImageDataGenerator(rescale=1./255)

# Load training and validation datasets
train_generator = datagen.flow_from_directory(
    '../data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

validation_generator = datagen.flow_from_directory(
    '../data/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')
