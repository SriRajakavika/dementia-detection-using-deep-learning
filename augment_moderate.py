from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img, save_img
import os

# Paths
moderate_dir = 'dataset/ModerateDemented'
augmented_dir = 'dataset/ModerateDemented_augmented'

if not os.path.exists(augmented_dir):
    os.makedirs(augmented_dir)

# Data augmentation settings
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Augment images
for img_name in os.listdir(moderate_dir):
    img_path = os.path.join(moderate_dir, img_name)
    img = load_img(img_path)  # load image
    x = img_to_array(img)      # convert to array
    x = x.reshape((1,) + x.shape)

    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=augmented_dir,
                              save_prefix='aug', save_format='jpg'):
        i += 1
        if i >= 10:  # generate 10 augmented images per original
            break

print("✅ Augmentation done!")
