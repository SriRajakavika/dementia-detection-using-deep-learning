import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from glob import glob
import random

# Paths
original_path = "dataset/ModerateDemented"  # original images
train_path = "dataset/train/ModerateDemented"
val_path = "dataset/val/ModerateDemented"

# Create folders if not exist
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)

# Parameters
num_augmented_per_image = 5  # how many augmented images per original image

# ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Get all original images
images = glob(os.path.join(original_path, "*.jpg")) + glob(os.path.join(original_path, "*.png"))
random.shuffle(images)

# Split 80% train, 20% val
split_idx = int(len(images) * 0.8)
train_imgs = images[:split_idx]
val_imgs = images[split_idx:]

def augment_and_save(img_list, dest_folder, prefix):
    count = 1
    for img_path in img_list:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        x = np.expand_dims(img, 0)
        for batch in datagen.flow(x, batch_size=1):
            new_name = f"{prefix}_{count}.jpg"
            cv2.imwrite(os.path.join(dest_folder, new_name), batch[0])
            count += 1
            if count % num_augmented_per_image == 0:
                break

# Augment and save
augment_and_save(train_imgs, train_path, "moderate_aug_train")
augment_and_save(val_imgs, val_path, "moderate_aug_val")

print("✅ Augmentation done and images saved directly to train/val with unique names!")
