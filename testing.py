from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create ImageDataGenerator (same normalization used during training)
datagen = ImageDataGenerator(rescale=1./255)

# Create validation generator
val_generator = datagen.flow_from_directory(
    r"D:\Project\neurocognet\dataset\val",  # 🔁 CHANGE to your actual validation folder path
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)
# Load model
model = load_model("alzheimer_model_cpu.h5")

# Recreate validation generator here
# (same preprocessing settings you used during training)

# Then evaluate
loss, accuracy = model.evaluate(val_generator)
print("Validation Accuracy: {:.2f}%".format(accuracy * 100))

y_pred = model.predict(val_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

print(classification_report(val_generator.classes, y_pred_classes))