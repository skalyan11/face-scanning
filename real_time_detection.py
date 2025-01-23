import cv2
import tensorflow as tf
import numpy as np

# Constants
IMG_WIDTH = 750
IMG_HEIGHT = 540

# Load the trained model
print("Loading model...")
model = tf.keras.models.load_model('facetracker.h5')

# Initialize video capture
cap = cv2.VideoCapture(1)  # Try 0 if 1 doesn't work

while cap.isOpened():
    _, frame = cap.read()
    
    # Get frame from center of the image
    frame = frame[50:500, 50:500,:]
    
    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize and preprocess for model
    resized = tf.image.resize(rgb, (IMG_HEIGHT, IMG_WIDTH))
    resized = resized / 255.0
    
    # Add batch dimension
    yhat = model.predict(np.expand_dims(resized, 0))
    sample_coords = yhat[1][0]
    
    # Check confidence score
    if yhat[0] > 0.5:
        # Controls the main rectangle
        start_point = tuple(np.multiply(sample_coords[:2], [450,450]).astype(int))
        end_point = tuple(np.multiply(sample_coords[2:], [450,450]).astype(int))
        cv2.rectangle(frame, start_point, end_point, (255,0,0), 2)
        
        # Controls the label rectangle
        cv2.rectangle(frame, 
                     tuple(np.add(start_point, [0,-30])),
                     tuple(np.add(start_point, [80,0])),
                     (255,0,0), -1)
        
        # Controls the text rendered
        cv2.putText(frame, 'face', tuple(np.add(start_point, [0,-5])),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    
    cv2.imshow('EyeTrack', frame)
    
    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows() 