from classificationModel import model
from loadImages import train_generator,validation_generator
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)
