import librosa

def extract_audio_features(file, n_mfcc=20, n_fft=2048, hop_length=512):
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
