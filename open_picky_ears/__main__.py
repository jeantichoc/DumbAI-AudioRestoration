import os
import numpy as np
import tensorflow as tf

from open_picky_ears.audio_features_extractor import extract_audio_features

n_mfcc = 20


def list_wav(input_dir_path):
    wav_files = []
    for root, dirs, files in os.walk(input_dir_path):
        for file in files:
            # check if the file is an audio file
            if file.endswith(".wav") or file.endswith(".flac"):
                wav_files.append(os.path.join(root, file))
    wav_files.sort()
    return wav_files


def load_data(path):
    features = []
    for filename in list_wav(path):
        print("loading " + filename)
        feature = extract_audio_features(filename, n_mfcc=n_mfcc)
        features.append(feature)
    return np.array(features)


# Load the input and target data
input_data = load_data("../prepared_input")
target_data = load_data("../prepared_target")
target_data = np.expand_dims(target_data, axis=-1)  # Add an additional dimension

# Define the neural network architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(862, n_mfcc)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')  # Change the output dimension
])

# Compile the model with a loss function and optimizer
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(input_data, target_data, epochs=100, batch_size=32)

# Test the model on some new data
test_input_data = np.load("test_input.npy")
predictions = model.predict(test_input_data)

# Save the predictions
np.save("predictions.npy", predictions)
