import json
import tensorflow as tf

def load_labels(label_path):
    with open(label_path.numpy(), 'r', encoding='utf-8') as f:
        label = json.load(f)
    # Ensure class is a single float32 value and bbox is a float32 array
    class_label = tf.cast(label['class'], tf.float32)
    bbox = tf.cast(label['bbox'], tf.float32)
    return [class_label, bbox]

train_labels = tf.data.Dataset.list_files('aug_data/train/labels/*.json', shuffle=False)
train_labels = train_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.float32, tf.float32]))

test_labels = tf.data.Dataset.list_files('aug_data/test/labels/*.json', shuffle=False)
test_labels = test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.float32, tf.float32]))

val_labels = tf.data.Dataset.list_files('aug_data/val/labels/*.json', shuffle=False)
val_labels = val_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.float32, tf.float32]))

# Ensure shapes are set explicitly
train_labels = train_labels.map(lambda c, b: (tf.ensure_shape(c, []), tf.ensure_shape(b, [4])))
test_labels = test_labels.map(lambda c, b: (tf.ensure_shape(c, []), tf.ensure_shape(b, [4])))
val_labels = val_labels.map(lambda c, b: (tf.ensure_shape(c, []), tf.ensure_shape(b, [4]))) 