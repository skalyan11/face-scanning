import tensorflow as tf
import json
import numpy as np
from matplotlib import pyplot as plt
import os
import cv2
import albumentations as alb
import os
import json

# Enable mixed precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Constants for optimization
IMG_HEIGHT = 540  # Reduced from 1080
IMG_WIDTH = 750   # Reduced from 1500
BATCH_SIZE = 32   # Increased from 16
BUFFER_SIZE = 1000

def load_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img

# Ensure GPU memory growth is set correctly
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# Cache the datasets after preprocessing
train_images = tf.data.Dataset.list_files('aug_data/train/images/*.jpg', shuffle=False)
train_images = train_images.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
train_images = train_images.map(lambda x: tf.image.resize(x, (IMG_HEIGHT, IMG_WIDTH)), num_parallel_calls=tf.data.AUTOTUNE)
train_images = train_images.map(lambda x: x/255.0, num_parallel_calls=tf.data.AUTOTUNE)
train_images = train_images.cache()

test_images = tf.data.Dataset.list_files('aug_data/test/images/*.jpg', shuffle=False)
test_images = test_images.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_images = test_images.map(lambda x: tf.image.resize(x, (IMG_HEIGHT, IMG_WIDTH)), num_parallel_calls=tf.data.AUTOTUNE)
test_images = test_images.map(lambda x: x/255.0, num_parallel_calls=tf.data.AUTOTUNE)
test_images = test_images.cache()

val_images = tf.data.Dataset.list_files('aug_data/val/images/*.jpg', shuffle=False)
val_images = val_images.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
val_images = val_images.map(lambda x: tf.image.resize(x, (IMG_HEIGHT, IMG_WIDTH)), num_parallel_calls=tf.data.AUTOTUNE)
val_images = val_images.map(lambda x: x/255.0, num_parallel_calls=tf.data.AUTOTUNE)
val_images = val_images.cache()

def load_labels(label_path):
    with open(label_path.numpy(), 'r', encoding='utf-8') as f:
        label = json.load(f)
    return [label['class'], label['bbox']]

# Optimize label loading
train_labels = tf.data.Dataset.list_files('aug_data/train/labels/*.json', shuffle=False)
train_labels = train_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float32]), 
                              num_parallel_calls=tf.data.AUTOTUNE)
train_labels = train_labels.cache()

test_labels = tf.data.Dataset.list_files('aug_data/test/labels/*.json', shuffle=False)
test_labels = test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float32]),
                            num_parallel_calls=tf.data.AUTOTUNE)
test_labels = test_labels.cache()

val_labels = tf.data.Dataset.list_files('aug_data/val/labels/*.json', shuffle=False)
val_labels = val_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float32]),
                          num_parallel_calls=tf.data.AUTOTUNE)
val_labels = val_labels.cache()

#print(train_labels.as_numpy_iterator().next())

#print(len(train_images), len(train_labels), len(test_images), len(test_labels), len(val_images), len(val_labels))


# Optimize final dataset preparation
train = tf.data.Dataset.zip((train_images, train_labels))
train = train.shuffle(BUFFER_SIZE)
train = train.batch(BATCH_SIZE)
train = train.prefetch(tf.data.AUTOTUNE)

test = tf.data.Dataset.zip((test_images, test_labels))
test = test.shuffle(BUFFER_SIZE)
test = test.batch(BATCH_SIZE)
test = test.prefetch(tf.data.AUTOTUNE)

val = tf.data.Dataset.zip((val_images, val_labels))
val = val.shuffle(BUFFER_SIZE)
val = val.batch(BATCH_SIZE)
val = val.prefetch(tf.data.AUTOTUNE)

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
    input_layer = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    # Use VGG16 with weights and make early layers non-trainable
    vgg = VGG16(include_top=False, input_tensor=input_layer, weights='imagenet')
    for layer in vgg.layers[:15]:  # Freeze early layers
        layer.trainable = False
    
    # Use shared features to reduce computation
    shared_features = GlobalMaxPooling2D()(vgg.output)
    
    # Classification branch
    class1 = Dense(1024, activation='relu')(shared_features)  # Reduced from 2048
    class2 = Dense(1, activation='sigmoid')(class1)
    
    # Regression branch
    regress1 = Dense(1024, activation='relu')(shared_features)  # Reduced from 2048
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

# Training configuration
batches_per_epoch = len(train)
lr_decay = (1.0 / 0.75 - 1.0) / batches_per_epoch

# Use a learning rate schedule instead of decay
initial_learning_rate = 0.0001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=batches_per_epoch,
    decay_rate=0.9,
    staircase=True
)

opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Create callbacks
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir="logs/facetracker",
    histogram_freq=1,
    profile_batch='500,520'
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_total_loss',
    mode='min',
    patience=5,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_total_loss',
    mode='min',
    factor=0.2,
    patience=3,
    min_lr=1e-6
)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.weights.h5',
    monitor='val_total_loss',
    mode='min',
    save_best_only=True,
    save_weights_only=True
)

callbacks = [
    tensorboard_callback,
    early_stopping,
    reduce_lr,
    model_checkpoint
]

#create localization loss
def localization_loss(y_true, y_pred):
    # Ensure both inputs are float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Scale factor to balance with classification loss
    scale_factor = 0.1
    
    # Coordinate loss
    delta_coord = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Size loss
    h_true = y_true[:, 2] - y_true[:, 0]
    w_true = y_true[:, 3] - y_true[:, 1]
    
    h_pred = y_pred[:, 2] - y_pred[:, 0]
    w_pred = y_pred[:, 3] - y_pred[:, 1]
    
    delta_size = tf.reduce_mean(tf.square(h_true - h_pred) + tf.square(w_true - w_pred))
    
    # Apply scaling to balance with classification loss
    return scale_factor * (delta_coord + delta_size)

classloss = tf.keras.losses.BinaryCrossentropy()
regressloss = localization_loss

print(localization_loss(y[1], coords))
print(classloss(y[0], classes))


class FaceTracker(Model):
    def __init__(self, eyetracker, **kwargs):
        super().__init__(**kwargs)
        self.model = eyetracker
    
    def compile(self, opt, classloss, localizationloss, **kwargs):
        super().compile(**kwargs)
        self.opt = opt
        self.closs = classloss
        self.lloss = localizationloss
    
    @tf.function
    def train_step(self, batch):
        x, y = batch
        
        with tf.GradientTape() as tape:
            # Forward pass
            classes, coords = self(x, training=True)
            
            # Reshape class predictions and labels for loss calculation
            classes = tf.reshape(classes, [-1])
            y_class = tf.reshape(y[0], [-1])
            
            # Calculate losses
            batch_classloss = self.closs(y_class, classes)
            batch_regloss = self.lloss(y[1], coords)
            total_loss = batch_classloss + batch_regloss
            
        # Gradient calculation and optimization
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.opt.apply_gradients(zip(gradients, self.trainable_variables))
            
        return {"total_loss": total_loss, "class_loss": batch_classloss, "regress_loss": batch_regloss}
    
    @tf.function
    def test_step(self, batch):
        x, y = batch
        classes, coords = self(x, training=False)
        
        # Reshape predictions and labels
        classes = tf.reshape(classes, [-1])
        y_class = tf.reshape(y[0], [-1])
        
        # Calculate losses
        batch_classloss = self.closs(y_class, classes)
        batch_regloss = self.lloss(y[1], coords)
        total_loss = batch_classloss + batch_regloss
        
        return {"total_loss": total_loss, "class_loss": batch_classloss, "regress_loss": batch_regloss}
    
    def call(self, x, **kwargs):
        return self.model(x, **kwargs)
    
    

model = FaceTracker(facetracker)
model.compile(opt, classloss, regressloss)

hist = model.fit(
    train,
    validation_data=val,
    epochs=20,
    callbacks=callbacks
)

# Save training history
with open('training_history.json', 'w') as f:
    json.dump(hist.history, f)

# Make predictions on test set
print("\nMaking predictions on test set...")
test_data = test.as_numpy_iterator()
test_sample = test_data.next()

yhat = facetracker.predict(test_sample[0])

# Visualize predictions
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx in range(4):
    sample_image = test_sample[0][idx].copy()
    sample_coords = yhat[1][idx]
    
    if yhat[0][idx] > 0.5:  # Threshold for face detection
        # Scale coordinates to image dimensions
        start_point = tuple(np.multiply(sample_coords[:2], [IMG_WIDTH, IMG_HEIGHT]).astype(int))
        end_point = tuple(np.multiply(sample_coords[2:], [IMG_WIDTH, IMG_HEIGHT]).astype(int))
        
        # Draw rectangle
        cv2.rectangle(
            img=sample_image,
            pt1=start_point,
            pt2=end_point,
            color=(0,255,0),  # Green color for predictions
            thickness=2
        )
    
    ax[idx].imshow(sample_image)
    ax[idx].axis('off')

plt.show()

# Save the model
print("\nSaving model...")
facetracker.save('facetracker_base.h5')  # Save the base model instead of the wrapper

# No need to test load here
print("Model saved successfully!")

