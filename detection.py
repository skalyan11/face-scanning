import tensorflow as tf
import json
import numpy as np
from matplotlib import pyplot as plt
import os
import cv2
import albumentations as alb
import os

# Ensure GPU memory growth is set correctly
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Check if the images path is correct
images = tf.data.Dataset.list_files('data/images/*.jpg', shuffle=False)
#print(images.as_numpy_iterator().next())

def load_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img

images = images.map(load_image)
image_generator = images.batch(4).as_numpy_iterator()

# Uncomment and use this block to visualize images if needed
# while True:
#     plot_images = image_generator.next()
#     fig, ax = plt.subplots(ncols=3, figsize=(20, 20))
#     for idx, image in enumerate(plot_images):
#         ax[idx].imshow(image)
#     plt.show()
#     input("Press Enter to load the next batch...")

# Define augmentations
augmentations = alb.Compose([alb.RandomCrop(1080, 1500), 
                             alb.HorizontalFlip(p=0.5), 
                             alb.RandomBrightnessContrast(p=0.2),
                             alb.RandomGamma(p=0.2),
                             alb.RGBShift(p=0.2),
                             alb.VerticalFlip(p=0.5)],
                            bbox_params = alb.BboxParams(format='albumentations', label_fields = ['class_labels']))

# Test the augmentation pipeline
img_path = '/Users/srikumarkalyan/Personal/FacialRecognition/face-scanning/data/train/images/85c93c72-d2cf-11ef-9f31-0674f0303784.jpg'
json_path = '/Users/srikumarkalyan/Personal/FacialRecognition/face-scanning/data/train/labels/85c93c72-d2cf-11ef-9f31-0674f0303784.json'

# Ensure the image and JSON file paths are correct
if os.path.exists(img_path) and os.path.exists(json_path):
    img = cv2.imread(img_path)
    with open(json_path) as f:
        label = json.load(f)
    #print(label)
else:
    print(f"File not found: {img_path} or {json_path}")

# Extract and normalize coordinates
coords = [point for shape in label['shapes'] for point in shape['points']]
coords = [coords[0][0], coords[0][1], coords[1][0], coords[1][1]]

# Normalize coordinates to be between 0 and 1
coords = list(np.divide(coords, [label['imageWidth'], label['imageHeight'], label['imageWidth'], label['imageHeight']]))

# Apply augmentations
augmented = augmentations(image=img, bboxes=[coords], class_labels=['face'])

# Convert normalized coordinates back to pixel values for drawing
height, width = img.shape[:2]
start_point = tuple(np.multiply(augmented['bboxes'][0][:2], [width, height]).astype(int))
end_point = tuple(np.multiply(augmented['bboxes'][0][2:], [width, height]).astype(int))

# Draw the bounding box on the augmented image
cv2.rectangle(augmented['image'], start_point, end_point, (255, 0, 0), 2)

# Display the image with the bounding box
plt.imshow(cv2.cvtColor(augmented['image'], cv2.COLOR_BGR2RGB))
#plt.show()





IMAGE_DIR = '/Users/srikumarkalyan/Personal/FacialRecognition/face-scanning/data/images'

# Verify directory exists
if not os.path.exists(IMAGE_DIR):
    raise Exception(f"Directory not found: {IMAGE_DIR}")

# List all jpg images in directory
images = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]


#load the training data
for partition in ['train','test','val']: 
    for image in images:
        img = cv2.imread(os.path.join('data', partition, 'images', image))

        coords = [0,0,0.00001,0.00001]
        label_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', partition, 'labels', f'{image.split(".")[0]}.json')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label = json.load(f)

            coords[0] = label['shapes'][0]['points'][0][0]
            coords[1] = label['shapes'][0]['points'][0][1]
            coords[2] = label['shapes'][0]['points'][1][0]
            coords[3] = label['shapes'][0]['points'][1][1]
            height, width = img.shape[:2]
            coords = list(np.divide(coords, [width,height,width,height]))

        try: 
            for x in range(60):
                augmented = augmentations(image=img, bboxes=[coords], class_labels=['face'])
                cv2.imwrite(os.path.join('aug_data', partition, 'images', f'{image.split(".")[0]}.{x}.jpg'), augmented['image'])

                annotation = {}
                annotation['image'] = image

                if os.path.exists(label_path):
                    if len(augmented['bboxes']) == 0: 
                        annotation['bbox'] = [0,0,0,0]
                        annotation['class'] = 0 
                    else: 
                        annotation['bbox'] = augmented['bboxes'][0]
                        annotation['class'] = 1
                else: 
                    annotation['bbox'] = [0,0,0,0]
                    annotation['class'] = 0 


                with open(os.path.join('aug_data', partition, 'labels', f'{image.split(".")[0]}.{x}.json'), 'w') as f:
                    json.dump(annotation, f)

        except Exception as e:
            print(e)
    
    
    
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

