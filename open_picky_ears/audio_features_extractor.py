import librosa
import librosa.display
import matplotlib.pyplot as plt


def extract_audio_features(file, n_mfcc=60, n_fft=2048, hop_length=512):
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
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)

    return mfccs


def visualize_data(mfccs):
    # Plot MFCCs
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    librosa.display.specshow(mfccs)
    plt.ylabel('MFCC')

    plt.tight_layout()
    plt.show()


def visualize_data_2(mfccs1, mfccs2):
    # Plot MFCCs
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    librosa.display.specshow(mfccs1)
    plt.ylabel('degraded')

    plt.subplot(2, 1, 2)
    librosa.display.specshow(mfccs1)
    plt.ylabel('restored')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    input_file_path = "../lossy_audio.wav"
    data = extract_audio_features(input_file_path)
    visualize_data_2(data, data)
