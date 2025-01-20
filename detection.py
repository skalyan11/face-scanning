import tensorflow as tf
import json
import numpy as np
from matplotlib import pyplot as plt
import os
import cv2
import albumentations as alb
import os
from image_into_data import load_image
from image_into_data import images

# Ensure GPU memory growth is set correctly
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


#load augmented images to tensorflow dataset
train_images = tf.data.Dataset.list_files('aug_data/train/images/*.jpg', shuffle=False)
train_images = images.map(load_image)
train_images = train_images.map(lambda x: tf.image.resize(x, (1080, 1500)))
train_images = train_images.map(lambda x: x/255.0)

test_images = tf.data.Dataset.list_files('aug_data/test/images/*.jpg', shuffle=False)
test_images = images.map(load_image)
test_images = test_images.map(lambda x: tf.image.resize(x, (1080, 1500)))
test_images = test_images.map(lambda x: x/255.0)

val_images = tf.data.Dataset.list_files('aug_data/val/images/*.jpg', shuffle=False)
val_images = images.map(load_image)
val_images = images.map(lambda x: tf.image.resize(x, (1080, 1500)))
val_images = images.map(lambda x: x/255.0)

