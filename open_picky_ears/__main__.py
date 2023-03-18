import os
import numpy as np
import tensorflow as tf

from open_picky_ears.audio_features_extractor import extract_audio_features

n_mfcc = 20
model_dir = "../trained_model"

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

print("input shape = " + str(input_data.shape))
print("target shape = " + str(target_data.shape))

input_shape = input_data.shape[1:]

# Define the neural network architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')  # Change the output dimension
])

# Compile the model with a loss function and optimizer
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(input_data, target_data, epochs=10, batch_size=8, validation_split=0.2)

model.save(model_dir)
