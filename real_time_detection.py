import cv2
import tensorflow as tf
import numpy as np

# Constants
IMG_WIDTH = 750
IMG_HEIGHT = 540

# Load the trained model
print("Loading model...")
model = tf.keras.models.load_model('facetracker_base.h5')

# Initialize video capture (try different indices if one doesn't work)
cap = cv2.VideoCapture(0)  # Try 0 first, then 1 if it doesn't work
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("Starting video capture... Press 'q' to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break
        
    # Get frame from center of the image and maintain aspect ratio
    h, w = frame.shape[:2]
    min_dim = min(h, w)
    start_y = (h - min_dim) // 2
    start_x = (w - min_dim) // 2
    frame = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
    
    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize and preprocess for model
    resized = tf.image.resize(rgb, (IMG_HEIGHT, IMG_WIDTH))
    resized = resized / 255.0
    
    # Add batch dimension and make prediction
    yhat = model.predict(np.expand_dims(resized, 0), verbose=0)
    sample_coords = yhat[1][0]
    confidence = yhat[0][0]
    
    # Check confidence score
    if confidence > 0.5:
        # Scale coordinates back to frame size
        frame_size = frame.shape[0]
        start_point = tuple(np.multiply(sample_coords[:2], [frame_size, frame_size]).astype(int))
        end_point = tuple(np.multiply(sample_coords[2:], [frame_size, frame_size]).astype(int))
        
        # Draw main rectangle
        cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)
        
        # Draw label background
        cv2.rectangle(frame, 
                     tuple(np.add(start_point, [0, -30])),
                     tuple(np.add(start_point, [120, 0])),
                     (0, 255, 0), -1)
        
        # Add confidence score to label
        label = f'Face: {confidence:.2f}'
        cv2.putText(frame, label, 
                   tuple(np.add(start_point, [0, -5])),
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 255, 255), 2, 
                   cv2.LINE_AA)
    
    # Show the frame
    cv2.imshow('Face Tracker', frame)
    
    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
print("Video capture ended") 