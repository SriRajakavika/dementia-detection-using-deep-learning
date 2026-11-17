import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import os

# ------------------------
# Load model
# ------------------------
model = tf.keras.models.load_model("alzheimer_model_cpu.h5")

# ------------------------
# Dataset path
# ------------------------
dataset_path = r"D:\Project\neurocognet\dataset"
val_dir = os.path.join(dataset_path, "val")

img_height, img_width = 224, 224
batch_size = 16

val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# ------------------------
# Evaluate model
# ------------------------
loss, accuracy = model.evaluate(val_generator)
print("\nTest Accuracy:", round(accuracy * 100, 2), "%")

# Predictions
pred = model.predict(val_generator)
predicted_classes = np.argmax(pred, axis=1)
true_classes = val_generator.classes

print("\nClassification Report:\n")
print(classification_report(true_classes, predicted_classes))

print("\nConfusion Matrix:\n")
print(confusion_matrix(true_classes, predicted_classes))
