import os
import numpy as np
from keras import layers, models

from dumb_audio_restoration.audio_features_extractor import wav_to_array


def list_wav(input_dir_path):
    wav_files = []
    for root, dirs, files in os.walk(input_dir_path):
        for file in files:
            # check if the file is an audio file
            if file.endswith(".wav"):
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


def train_model():
    # Load the input and target data
    input_data = load_data("../prepared_inputs")
    target_data = load_data("../prepared_targets")
    print("input shape = " + str(input_data.shape))
    print("target shape = " + str(target_data.shape))
    input_shape = input_data.shape[1:]
    # Define the neural network architecture
    model = models.Sequential([
        layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(44100, 1)),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(44100)
    ])
    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    # Print a summary of the model architecture
    model.summary()
    # Train the model
    model.fit(input_data, target_data, epochs=100, batch_size=8, validation_split=0.2)
    return model


if __name__ == '__main__':
    trained_model = train_model()
    trained_model.save("../trained_model")
