import os
import numpy as np
from keras import layers, models

from open_picky_ears.audio_features_extractor import extract_audio_features
from open_picky_ears.new_audio_features_extractor import wav_to_array

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
        feature = wav_to_array(filename)
        features.append(feature)
    return np.array(features)


# Load the input and target data
input_data = load_data("../prepared_input")
target_data = load_data("../prepared_target")

print("input shape = " + str(input_data.shape))
print("target shape = " + str(target_data.shape))

input_shape = input_data.shape[1:]

# Define the neural network architecture
model = models.Sequential([
    layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(441000, 1)),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(441000)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Print a summary of the model architecture
model.summary()
# Train the model
model.fit(input_data, target_data, epochs=20, batch_size=32, validation_split=0.2)

model.save(model_dir)
