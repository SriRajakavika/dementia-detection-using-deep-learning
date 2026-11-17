from tensorflow.keras.models import load_model

model = load_model("alzheimer_model_cpu.h5")
print("✅ Alzheimer CPU model loaded successfully!")
model.summary()
