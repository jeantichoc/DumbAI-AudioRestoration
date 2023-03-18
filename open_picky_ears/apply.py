import tensorflow as tf
import librosa
import numpy as np
import soundfile as sf
from scipy.fft import idct

from open_picky_ears.audio_features_extractor import extract_audio_features

model_filename = "../trained_model"


def apply_model_on_file(input_file, output_file, model_file=model_filename):
    # Load the pre-trained TensorFlow model
    model = tf.keras.models.load_model(model_file)

    signal, sr = librosa.load(input_file)

    input_features = extract_audio_features(input_file, n_mfcc=60)

    # Reshape the input features to match the expected shape of the model's input
    input_features = np.reshape(input_features, (1, *input_features.shape))

    # Use the model to predict the output
    predicted_output = model.predict(input_features)

    # Print the predicted output
    print("Predicted output:", predicted_output)

    # Reshape the predicted output to match the shape of the original features
    predicted_output = np.reshape(predicted_output, input_features.shape[1:])

    # Invert the MFCCs using the IDCT to resynthesize the audio waveform
    reconstructed_audio = idct(predicted_output, type=2, axis=0)

    # Save the reconstructed audio waveform to a new WAV file
    sf.write(output_file, reconstructed_audio, int(sr))




if __name__ == '__main__':
    input_file_path = "../lossy_audio.wav"
    apply_model_on_file(input_file_path, "../new.wav")
