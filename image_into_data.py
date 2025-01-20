import os
import json
import cv2
import numpy as np
import albumentations as alb
import tensorflow as tf
from matplotlib import pyplot as plt




gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Check if the images path is correct
images = tf.data.Dataset.list_files('data/images/*.jpg', shuffle=False)
print(images.as_numpy_iterator().next())
#print(images.as_numpy_iterator().next())

def load_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img

images = images.map(load_image)
image_generator = images.batch(4).as_numpy_iterator()
'''
# Visualize images
while True:
    plot_images = image_generator.next()
    fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
    for idx, image in enumerate(plot_images):
        ax[idx].imshow(image)
    plt.show()
    input("Press Enter to load the next batch...")
'''
# Split images into train, test, and validation sets
# Get total number of images first
total_images = sum(1 for _ in images)

# Calculate splits based on actual image count
train_size = int(total_images * 0.7)
test_size = int(total_images * 0.2)
val_size = total_images - (train_size + test_size)  # Remaining images

# Reset the iterator
images = tf.data.Dataset.list_files('data/images/*.jpg')

# Split according to actual counts
train_images = images.take(train_size)
test_images = images.skip(train_size).take(test_size)
val_images = images.skip(train_size + test_size).take(val_size)

# Save images into respective folders
train_images = train_images.as_numpy_iterator()
test_images = test_images.as_numpy_iterator()
val_images = val_images.as_numpy_iterator()

# Process train images and labels
for image_path in train_images:
    try:
        # Convert bytes to string for path handling
        image_path_str = image_path.decode() if isinstance(image_path, bytes) else str(image_path)
        
        # Get original filename without extension
        original_name = os.path.basename(image_path_str)
        base_name = os.path.splitext(original_name)[0]
        
        # Process image
        image = tf.io.read_file(image_path_str)
        image = tf.image.decode_jpeg(image)
        image = image.numpy()
        cv2.imwrite(f'data/train/images/{original_name}', image)
        
        # Copy JSON with original filename
        json_path = os.path.join('data/labels', f'{base_name}.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as src:
                json_content = json.load(src)
                # Update image path in JSON to maintain relative path
                json_content['imagePath'] = f'../images/{original_name}'
            with open(f'data/train/labels/{base_name}.json', 'w') as dst:
                json.dump(json_content, dst, indent=4)
    except Exception as e:
        print(f"Error processing file {original_name}: {e}")

# Process test images and labels
for image_path in test_images:
    try:
        image_path_str = image_path.decode() if isinstance(image_path, bytes) else str(image_path)
        
        # Get original filename without extension
        original_name = os.path.basename(image_path_str)
        base_name = os.path.splitext(original_name)[0]
        
        # Process image
        image = tf.io.read_file(image_path_str)
        image = tf.image.decode_jpeg(image)
        image = image.numpy()
        cv2.imwrite(f'data/test/images/{original_name}', image)
        
        # Copy JSON with original filename
        json_path = os.path.join('data/labels', f'{base_name}.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as src:
                json_content = json.load(src)
                # Update image path in JSON to maintain relative path
                json_content['imagePath'] = f'../images/{original_name}'
            with open(f'data/test/labels/{base_name}.json', 'w') as dst:
                json.dump(json_content, dst, indent=4)
    except Exception as e:
        print(f"Error processing file {original_name}: {e}")

# Process validation images and labels
for image_path in val_images:
    try:
        image_path_str = image_path.decode() if isinstance(image_path, bytes) else str(image_path)
        
        # Get original filename without extension
        original_name = os.path.basename(image_path_str)
        base_name = os.path.splitext(original_name)[0]
        
        # Process image
        image = tf.io.read_file(image_path_str)
        image = tf.image.decode_jpeg(image)
        image = image.numpy()
        cv2.imwrite(f'data/val/images/{original_name}', image)
        
        # Copy JSON with original filename
        json_path = os.path.join('data/labels', f'{base_name}.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as src:
                json_content = json.load(src)
                # Update image path in JSON to maintain relative path
                json_content['imagePath'] = f'../images/{original_name}'
            with open(f'data/val/labels/{base_name}.json', 'w') as dst:
                json.dump(json_content, dst, indent=4)
    except Exception as e:
        print(f"Error processing file {original_name}: {e}")

'''
# Get original JSON files
source_json_files = [f for f in os.listdir('data/labels') if f.endswith('.json')]

# Process JSON files for each split
for idx, image_path in enumerate(train_images):
    # Get image filename without extension
    image_name = os.path.splitext(os.path.basename(image_path.numpy().decode()))[0]
    # Find matching JSON
    json_file = f"{image_name}.json"
    if json_file in source_json_files:
        # Copy JSON content to new location
        with open(f'data/labels/{json_file}', 'r') as src:
            json_content = json.load(src)
        with open(f'data/train/labels/{idx}.json', 'w') as dst:
            json.dump(json_content, dst)

# Repeat for test_images
for idx, image_path in enumerate(test_images):
    image_name = os.path.splitext(os.path.basename(image_path.numpy().decode()))[0]
    json_file = f"{image_name}.json"
    if json_file in source_json_files:
        with open(f'data/labels/{json_file}', 'r') as src:
            json_content = json.load(src)
        with open(f'data/test/labels/{idx}.json', 'w') as dst:
            json.dump(json_content, dst)

# Repeat for val_images
for idx, image_path in enumerate(val_images):
    image_name = os.path.splitext(os.path.basename(image_path.numpy().decode()))[0]
    json_file = f"{image_name}.json"
    if json_file in source_json_files:
        with open(f'data/labels/{json_file}', 'r') as src:
            json_content = json.load(src)
        with open(f'data/val/labels/{idx}.json', 'w') as dst:
            json.dump(json_content, dst)
'''