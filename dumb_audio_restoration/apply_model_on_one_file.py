import tensorflow as tf
import numpy as np

from dumb_audio_restoration.audio_features_extractor import wav_to_array, array_to_wav

model_filename = "../trained_model"


def apply_model_on_file(input_file, output_file, model_file=model_filename):
    # Load the pre-trained TensorFlow model
    model = tf.keras.models.load_model(model_file)

    input_features = wav_to_array(input_file)

    # Reshape the input features to match the expected shape of the model's input
    input_features = input_features.T
    input_features = np.reshape(input_features, (1, *input_features.shape))
    print("input_features shape : " + str(input_features.shape))

    # Use the model to predict the output
    predicted_output = model.predict(input_features)

    print("predicted_output shape : " + str(predicted_output.shape))
    array_to_wav(predicted_output, output_file)


if __name__ == '__main__':
    input_file_path = "../tests/test.wav"
    apply_model_on_file(input_file_path, "../tests/restored.wav")
