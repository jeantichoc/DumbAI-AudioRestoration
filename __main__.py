import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow import keras


def list_wav(input_dir_path):
    wav_files = []
    for root, dirs, files in os.walk(input_dir_path):
        for file in files:
            # check if the file is an audio file
            if file.endswith(".wav") or file.endswith(".flac"):
                wav_files.append(os.path.join(root, file))
    wav_files.sort()
    return wav_files


# Step 1: Load and Prepare the Data
def extract_audio_features(file, num_mfcc=20, n_fft=2048, hop_length=512):
    """
    Extract audio features from WAV files and prepare them for TensorFlow.

    Args:
        file_path (str): Path to the directory containing the WAV files.
        num_mfcc (int): Number of Mel Frequency Cepstral Coefficients (MFCCs) to extract (default=13).
        n_fft (int): Number of points in the Fast Fourier Transform (FFT) (default=2048).
        hop_length (int): Number of samples between successive frames (default=512).
    """
    signal, sr = librosa.load(file)

    # Extract Mel Frequency Cepstral Coefficients (MFCCs) from audio signal
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=num_mfcc)

    return mfccs.T

def load_data(path):
    features = []
    for filename in list_wav(path):
        print("loading " + filename)
        feature = extract_audio_features(filename)
        features.append(feature)
    return np.array(features)


# Load the input and target data
input_data = load_data("prepared_input")
target_data = load_data("prepared_target")
target_data = np.expand_dims(target_data, axis=-1)  # Add an additional dimension

# Define the neural network architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(862, 20)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear') # Change the output dimension
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
