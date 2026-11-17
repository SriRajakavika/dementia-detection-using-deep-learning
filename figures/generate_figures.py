import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# -------------------------------
# 1️⃣ Load Trained Model
# -------------------------------
model = load_model("alzheimer_model_cpu.h5")

# -------------------------------
# 2️⃣ Create Validation Generator
# -------------------------------
datagen = ImageDataGenerator(rescale=1./255)

validation_generator = datagen.flow_from_directory(
    r"D:\Project\neurocognet\dataset\val",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# -------------------------------
# 3️⃣ Get Predictions
# -------------------------------
y_true = validation_generator.classes
y_pred_probs = model.predict(validation_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

# -------------------------------
# 4️⃣ Confusion Matrix
# -------------------------------
cm = confusion_matrix(y_true, y_pred)

print("Confusion Matrix:")
print(cm)

# -------------------------------
# 5️⃣ Plot Confusion Matrix
# -------------------------------
class_names = list(validation_generator.class_indices.keys())

plt.figure(figsize=(8,6))
sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("confusion_matrix.png")
plt.show()

# -------------------------------
# 6️⃣ Classification Report
# -------------------------------
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))
