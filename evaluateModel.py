from classificationModel import model
from loadImages import train_generator,validation_generator


test_loss, test_acc = model.evaluate(validation_generator)
print(f"Test accuracy: {test_acc}")
