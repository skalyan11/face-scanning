import json
import matplotlib.pyplot as plt
import os

# Check if history file exists
history_file = 'training_history.json'
if not os.path.exists(history_file):
    print(f"Error: {history_file} not found!")
    print("Please run program.py first to train the model and generate the training history.")
    exit()

# Load the training history
with open(history_file, 'r') as f:
    history = json.load(f)

# Create subplots for different metrics
plt.figure(figsize=(15, 10))

# Plot total loss
plt.subplot(2, 2, 1)
plt.plot(history['total_loss'], label='Training Total Loss')
plt.plot(history['val_total_loss'], label='Validation Total Loss')
plt.title('Total Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot classification loss
plt.subplot(2, 2, 2)
plt.plot(history['class_loss'], label='Training Class Loss')
plt.plot(history['val_class_loss'], label='Validation Class Loss')
plt.title('Classification Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot regression loss
plt.subplot(2, 2, 3)
plt.plot(history['regress_loss'], label='Training Regression Loss')
plt.plot(history['val_regress_loss'], label='Validation Regression Loss')
plt.title('Regression Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Print best values
print("\nBest Values:")
print(f"Best Total Loss: {min(history['val_total_loss']):.4f}")
print(f"Best Classification Loss: {min(history['val_class_loss']):.4f}")
print(f"Best Regression Loss: {min(history['val_regress_loss']):.4f}") 