
import albumentations as alb
from albumentations.core.composition import Compose
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

# Create augmentation without random cropping
augmentor = Compose([
    alb.HorizontalFlip(p=0.5)
], bbox_params=alb.BboxParams(format='albumentations', label_fields=['class_labels']))



### Validation checks ###

'''
IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'train', 'images')
print(IMAGE_DIR)

IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'test', 'images')
print(IMAGE_DIR)

IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'val', 'images')
print(IMAGE_DIR)


# Verify directory exists
if not os.path.exists(IMAGE_DIR):
    raise Exception(f"Directory not found: {IMAGE_DIR}")

# List all jpg images in directory
images = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
print(images)


JSON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'val', 'labels')
jsons = [os.path.join(JSON_DIR, f) for f in os.listdir(JSON_DIR) if f.endswith('.json')]
print(jsons)

'''
#load the data
for partition in ['train','test','val']: 
    for image in os.listdir(os.path.join('data', partition, 'images')):
        img_path = os.path.join('data', partition, 'images', image)
        img = cv2.imread(img_path)
        height, width = img.shape[:2]  # Get actual image dimensions
        label_path = os.path.join('data', partition, 'labels', image.split('.')[0] + '.json')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label = json.load(f)
        else:
            continue
        coords = [point for shape in label['shapes'] for point in shape['points']]
        x_min, y_min, x_max, y_max = coords[0][0], coords[0][1], coords[1][0], coords[1][1]
        
        coords = [x_min / label['imageWidth'], y_min / label['imageHeight'], x_max / label['imageWidth'], y_max / label['imageHeight']]
        try: 
            for x in range(3):
                augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])
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
