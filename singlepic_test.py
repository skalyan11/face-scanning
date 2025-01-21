import cv2
import os
import json
import numpy as np
from matplotlib import pyplot as plt
from albumentations.core.composition import Compose
import albumentations as alb

# Load image
img_path = os.path.join('data', 'train', 'images', '9d999f5e-d76b-11ef-af18-0674f0303784.jpg')
img = cv2.imread(img_path)
height, width = img.shape[:2]  # Get actual image dimensions

# Load JSON
with open(os.path.join('data', 'train', 'labels', '9d999f5e-d76b-11ef-af18-0674f0303784.json'), 'r') as f:
    label = json.load(f)

# Extract and normalize coordinates
coords = [point for shape in label['shapes'] for point in shape['points']]
x_min, y_min, x_max, y_max = coords[0][0], coords[0][1], coords[1][0], coords[1][1]

# Normalize coordinates
coords = [x_min / label['imageWidth'], y_min / label['imageHeight'], 
          x_max / label['imageWidth'], y_max / label['imageHeight']]

# Create augmentation without random cropping
augmentor = Compose([
    alb.HorizontalFlip(p=0.5)
], bbox_params=alb.BboxParams(format='albumentations', label_fields=['class_labels']))

# Apply augmentation using normalized coordinates
augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])

# Convert normalized coordinates back to pixel values for drawing
height, width = augmented['image'].shape[:2]
start_point = tuple(np.multiply(augmented['bboxes'][0][:2], [width, height]).astype(int))
end_point = tuple(np.multiply(augmented['bboxes'][0][2:], [width, height]).astype(int))

# Draw the bounding box on the augmented image
cv2.rectangle(augmented['image'], start_point, end_point, (250, 0, 0), 2)

# Display the image with the bounding box
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(augmented['image'], cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()