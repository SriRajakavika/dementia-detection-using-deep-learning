# ===============================
# train_model_pretrained.py
# Pretrained MobileNetV2 Fine-Tuning + Grad-CAM
# CPU-Friendly Version
# ===============================

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# -------------------------------
# 1️⃣ Paths to your dataset
# -------------------------------
train_dir = "dataset/train"
val_dir   = "dataset/val"
IMG_SIZE  = (224, 224)
BATCH_SIZE = 8

# -------------------------------
# 2️⃣ Data Augmentation
# -------------------------------
train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator()

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# -------------------------------
# 3️⃣ Load Pretrained MobileNetV2
# -------------------------------
base_model = MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)

# Freeze all layers initially
for layer in base_model.layers:
    layer.trainable = False

# -------------------------------
# 4️⃣ Add Custom Classifier
# -------------------------------
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
output = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# -------------------------------
# 5️⃣ Unfreeze Top 50 Layers
# -------------------------------
for layer in base_model.layers[-50:]:
    layer.trainable = True

# -------------------------------
# 6️⃣ Compile Model
# -------------------------------
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# 7️⃣ Callbacks
# -------------------------------
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=4,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

# -------------------------------
# 8️⃣ Train Model
# -------------------------------
EPOCHS = 20  # 15–20 epochs usually enough with pretrained
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr]
)

# -------------------------------
# 9️⃣ Save Fine-Tuned Model
# -------------------------------
model.save("mymobilenetv2_pretrained_finetuned.h5")
print("✅ Pretrained MobileNetV2 fine-tuned and saved!")

# -------------------------------
# 10️⃣ Grad-CAM Functions
# -------------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    plt.imshow(superimposed_img.astype(np.uint8))
    plt.axis('off')
    plt.show()