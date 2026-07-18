# convert_model.py
import sys
from unittest.mock import MagicMock

# Mock out libraries that are either not supported on Windows 
# or not needed for basic Keras conversion.
sys.modules['tensorflow_decision_forests'] = MagicMock()
sys.modules['wurlitzer'] = MagicMock()
sys.modules['jax'] = MagicMock()
sys.modules['jax.experimental'] = MagicMock()
sys.modules['flax'] = MagicMock()
sys.modules['flax.serialization'] = MagicMock()

import tensorflow as tf
import tensorflowjs as tfjs
import h5py
import numpy as np

h5_model_path = 'models/model.h5'
keras2_model_path = 'models/model_keras2.h5'

print("Rebuilding architecture using Keras 2 (tf_keras)...")
inputs = tf.keras.Input(shape=(32, 32, 3), name='input_1')

# Block 1
x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same', name='conv2d_1')(inputs)
x = tf.keras.layers.BatchNormalization(name='bn_1')(x)
x = tf.keras.layers.Conv2D(32, 3, activation='relu', name='conv2d_2')(x)
x = tf.keras.layers.MaxPooling2D(pool_size=2, name='pool_1')(x)
x = tf.keras.layers.Dropout(0.5, name='drop_1')(x)

# Block 2
x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', name='conv2d_3')(x)
x = tf.keras.layers.BatchNormalization(name='bn_2')(x)
x = tf.keras.layers.Conv2D(64, 3, activation='relu', name='conv2d_4')(x)
x = tf.keras.layers.MaxPooling2D(pool_size=2, name='pool_2')(x)
x = tf.keras.layers.Dropout(0.5, name='drop_2')(x)

# Block 3
x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', name='conv2d_5')(x)
x = tf.keras.layers.BatchNormalization(name='bn_3')(x)
x = tf.keras.layers.Conv2D(128, 3, activation='relu', name='conv2d_6')(x)
x = tf.keras.layers.MaxPooling2D(pool_size=2, name='pool_3')(x)
x = tf.keras.layers.Dropout(0.5, name='drop_3')(x)

# Flatten and Dense
x = tf.keras.layers.Flatten(name='flat')(x)
outputs = tf.keras.layers.Dense(3, activation='softmax', name='fc')(x)

keras2_model = tf.keras.Model(inputs=inputs, outputs=outputs, name='keras2_classifier')

print("Loading trained weights directly from HDF5 file...")
with h5py.File(h5_model_path, 'r') as f:
    weights_group = f['model_weights']
    
    # Map layers and assign weights
    # Conv2D layers: [kernel, bias]
    for conv_id in ['conv2d_1', 'conv2d_2', 'conv2d_3', 'conv2d_4', 'conv2d_5', 'conv2d_6']:
        kernel = weights_group[conv_id]['kernel'][:]
        bias = weights_group[conv_id]['bias'][:]
        keras2_model.get_layer(conv_id).set_weights([kernel, bias])
        print(f"  Loaded weights for {conv_id}")

    # BatchNormalization layers: [gamma, beta, moving_mean, moving_variance]
    for bn_id in ['bn_1', 'bn_2', 'bn_3']:
        gamma = weights_group[bn_id]['gamma'][:]
        beta = weights_group[bn_id]['beta'][:]
        moving_mean = weights_group[bn_id]['moving_mean'][:]
        moving_variance = weights_group[bn_id]['moving_variance'][:]
        keras2_model.get_layer(bn_id).set_weights([gamma, beta, moving_mean, moving_variance])
        print(f"  Loaded weights for {bn_id}")

    # Dense layer: [kernel, bias]
    kernel = weights_group['fc']['kernel'][:]
    bias = weights_group['fc']['bias'][:]
    keras2_model.get_layer('fc').set_weights([kernel, bias])
    print("  Loaded weights for fc (Dense)")

print("Saving the clean Keras 2 model...")
keras2_model.save(keras2_model_path)

# Verify the weight names are fully qualified in Keras 2
print("\nVerifying Keras 2 weight names:")
loaded_model = tf.keras.models.load_model(keras2_model_path, compile=False)
for w in loaded_model.weights:
    print(f"  Weight: {w.name}, shape: {w.shape}")

print("\nConverting clean Keras 2 model to TensorFlow.js format...")
tfjs.converters.save_keras_model(loaded_model, 'web_model')
print("Model converted successfully and saved to 'web_model/'!")
