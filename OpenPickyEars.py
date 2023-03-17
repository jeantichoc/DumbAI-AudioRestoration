import os
import random
import librosa
import numpy as np
import pydub
import tensorflow as tf
from keras.utils import pad_sequences
from tensorflow import keras


bitrates = ['64k', '96k', '128k']

def list_flac(lossless_dir):
    flac_files = []
    for root, dirs, files in os.walk(lossless_dir):
        for file in files:
            if file.lower().endswith('.flac'):
                flac_files.append(os.path.join(root, file))
    return flac_files

# Step 1: Load and Prepare the Data
def extract_audio_features_old(filename):
    y, sr = librosa.load(filename, sr=44100)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=16)
    return mfccs


def extract_audio_features(audio_file, maxlen=19220):
    y, sr = librosa.load(audio_file, sr=44100)
    y = librosa.util.normalize(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=256)
    delta = librosa.feature.delta(mfcc)
    features = np.concatenate((mfcc, delta), axis=0)

    # pad or truncate to fixed length
    if features.shape[1] < maxlen:
        features = np.pad(features, ((0, 0), (0, maxlen - features.shape[1])), 'constant')
    else:
        features = features[:, :maxlen]

    return features


def compress_audio(input_file, output_file):
    bitrate = random.choice(bitrates)
    pydub.AudioSegment.from_file(input_file).export(output_file, format="mp3", bitrate=f"{bitrate}")

def load_data():
    X_train = []
    Y_train = []

    for filename in list_flac("lossless_music"):
        print(filename)
        flac_file = filename
        mp3_file = "tmp.mp3"

        compress_audio(flac_file, mp3_file)
        X_train_features = extract_audio_features(mp3_file)
        Y_train_features = extract_audio_features(flac_file)

        print("X_train shape:", X_train_features.shape)
        print("Y_train shape:", Y_train_features.shape)

        X_train.append(X_train_features)
        Y_train.append(Y_train_features)

        os.remove(mp3_file)

    #X = pad_sequences(X_train, maxlen=20000, padding='post', truncating='post')
    #Y = pad_sequences(Y_train, maxlen=20000, padding='post', truncating='post')

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    return X_train, Y_train

X_train, Y_train = load_data()

# Step 2: Define the Neural Network Architecture
encoder = keras.models.Sequential([
    # keras.layers.Reshape((16, 32, 1), input_shape=[16]),
    keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(16, activation='relu'),
])

decoder = keras.models.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=[16]),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Reshape((8, 8, 4)),
    keras.layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same'),
    keras.layers.UpSampling2D((2, 2)),
    keras.layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same'),
    keras.layers.UpSampling2D((2, 2)),
    keras.layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same'),
])

autoencoder = keras.models.Sequential([encoder, decoder])

# Step 3: Train the Neural Network
X_train = tf.expand_dims(X_train, axis=-1)
Y_train = tf.expand_dims(Y_train, axis=-1)
X_val = X_train[:10]
Y_val = Y_train[:10]
X_train = X_train[10:]
Y_train = Y_train[10:]

#autoencoder.compile(loss='binary_crossentropy', optimizer='adam', run_eagerly=True)
autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'], run_eagerly=True)

history = autoencoder.fit(X_train, Y_train, epochs=50, batch_size=32, validation_data=(X_val, Y_val))

# Step 4: Evaluate the Model
X_test, Y_test = load_data()
X_test = tf.expand_dims(X_test, axis=-1)
Y_test = tf.expand_dims(Y_test, axis=-1)

test_loss = autoencoder.evaluate(X_test, Y_test)
print(f'Test Loss: {test_loss:.4f}')