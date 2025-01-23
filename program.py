import tensorflow as tf
import json
import numpy as np
from matplotlib import pyplot as plt
import os
import cv2
import albumentations as alb
import os
import json

def load_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img

# Ensure GPU memory growth is set correctly
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


#load augmented images to tensorflow dataset
train_images = tf.data.Dataset.list_files('aug_data/train/images/*.jpg', shuffle=False)
train_images = train_images.map(load_image)
train_images = train_images.map(lambda x: tf.image.resize(x, (1080, 1500)))
train_images = train_images.map(lambda x: x/255.0)

test_images = tf.data.Dataset.list_files('aug_data/test/images/*.jpg', shuffle=False)
test_images = test_images.map(load_image)
test_images = test_images.map(lambda x: tf.image.resize(x, (1080, 1500)))
test_images = test_images.map(lambda x: x/255.0)

val_images = tf.data.Dataset.list_files('aug_data/val/images/*.jpg', shuffle=False)
val_images = val_images.map(load_image)
val_images = val_images.map(lambda x: tf.image.resize(x, (1080, 1500)))
val_images = val_images.map(lambda x: x/255.0)

def load_labels(label_path):
    with open(label_path.numpy(), 'r', encoding='utf-8') as f:
        label = json.load(f)
    return [label['class'], label['bbox']]

train_labels = tf.data.Dataset.list_files('aug_data/train/labels/*.json', shuffle=False)
train_labels = train_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

test_labels = tf.data.Dataset.list_files('aug_data/test/labels/*.json', shuffle=False)
test_labels = test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

val_labels = tf.data.Dataset.list_files('aug_data/val/labels/*.json', shuffle=False)
val_labels = val_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

#print(train_labels.as_numpy_iterator().next())

#print(len(train_images), len(train_labels), len(test_images), len(test_labels), len(val_images), len(val_labels))


train = tf.data.Dataset.zip((train_images, train_labels))
train = train.shuffle(1000)
train = train.batch(8)
train = train.prefetch(4)

test = tf.data.Dataset.zip((test_images, test_labels))
test = test.shuffle(1000)
test = test.batch(8)
test = test.prefetch(4)


val = tf.data.Dataset.zip((val_images, val_labels))
val = val.shuffle(1000)
val = val.batch(8)
val = val.prefetch(4)

data_samples = train.as_numpy_iterator()
res = data_samples.next()

fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx in range(4): 
    # Convert to numpy and make writeable
    sample_image = res[0][idx].copy()  # Create writeable copy
    sample_coords = res[1][1][idx]
    
    # Get image dimensions
    height, width = sample_image.shape[:2]
    
    # Scale coordinates to image dimensions
    start_point = tuple(np.multiply(sample_coords[:2], [width, height]).astype(int))
    end_point = tuple(np.multiply(sample_coords[2:], [width, height]).astype(int))
    
    # Draw rectangle on copy
    cv2.rectangle(
        img=sample_image,
        pt1=start_point,
        pt2=end_point,
        color=(255,0,0),
        thickness=2
    )
    
    # Display
    ax[idx].imshow(sample_image)
    ax[idx].axis('off')

#plt.show()

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D
from tensorflow.keras.applications import VGG16

vgg = VGG16(include_top=False)

#vgg.summary()

def build_model():
    input_layer = Input(shape=(1080, 1500, 3))
    vgg = VGG16(include_top=False, input_tensor=input_layer)
    
    f1 = GlobalMaxPooling2D()(vgg.output)
    class1 = Dense(2048, activation='relu')(f1)
    class2 = Dense(1, activation='sigmoid')(class1)
    
    f2 = GlobalMaxPooling2D()(vgg.output)
    regress1 = Dense(2048, activation='relu')(f2)
    regress2 = Dense(4, activation='sigmoid')(regress1)
    
    facetracker = Model(inputs=input_layer, outputs=[class2, regress2])
    return facetracker

    
facetracker = build_model()
#facetracker.summary()


x, y = train.as_numpy_iterator().next()
print(x.shape)


classes, coords = facetracker.predict(x)

#print(classes, coords)

#slow down the learning to prevent overfitting
#print(len(train))

batches_per_epoch = len(train)
lr_decay = (1.0 / 0.75 - 1.0) / batches_per_epoch

opt = tf.keras.optimizers.Adam(learning_rate=0.0001, decay = lr_decay)

#create localization loss
def localization_loss(y_true, y_pred):
    delta_cord = tf.reduce_sum(tf.square(y_true - y_pred))
    
    h_true = y_true[:, 2] - y_true[:, 0]
    w_true = y_true[:, 3] - y_true[:, 1]
    
    h_pred = y_pred[:, 2] - y_pred[:, 0]
    w_pred = y_pred[:, 3] - y_pred[:, 1]
    
    delta_size = tf.reduce_sum(tf.square(h_true - h_pred) + tf.square(w_true - w_pred))
    
    return delta_cord + delta_size

classloss = tf.keras.losses.BinaryCrossentropy()
regressloss = localization_loss

print(localization_loss(y[1], coords))
print(classloss(y[0], classes))


class FaceTracker(Model):
    def __init__(self, eyetracker, **kwargs):
        super().__init__(**kwargs)
        self.model = eyetracker
    
    def compile(self, opt, classloss, localizationloss, **kwargs):
        super(FaceTracker, self).compile(**kwargs)
        self.opt = opt
        self.closs = classloss
        self.lloss = localizationloss
    
    def train_step(self, batch, **kwargs):
        x, y = batch
        
        with tf.GradientTape() as tape:
            classes, coords = self.model(x, training=True)
            batch_classloss = self.closs(y[0], classes)
            batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
            total_loss = batch_localizationloss + 0.5 * batch_classloss
            grad = tape.gradient(total_loss, self.model.trainable_variables)
        opt.apply_gradients(zip(grad, self.model.trainable_variables))
        return {'total_loss': total_loss, 'class_loss': batch_classloss, 'regress_loss': batch_localizationloss}
    
    def test_step(self, batch, **kwargs):
        x, y = batch
        classes, coords = self.model(x, training=False)
        batch_classloss = self.closs(y[0], classes)
        batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
        total_loss = batch_localizationloss + 0.5 * batch_classloss
        return {'total_loss': total_loss, 'class_loss': batch_classloss, 'regress_loss': batch_localizationloss}
    
        
    def call(self, x, **kwargs):
        return self.model(x, **kwargs)