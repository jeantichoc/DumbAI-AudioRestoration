import numpy as np
from pydub import AudioSegment


def wav_to_array(wav_file):
    sound = AudioSegment.from_wav(wav_file)
    sound = sound.set_channels(1)
    return np.array(sound.get_array_of_samples())


def array_to_wav(data, wav_file, sample_width=2, sample_rate=44100):
    sound = AudioSegment(
        data.tobytes(),
        sample_width=sample_width,
        frame_rate=sample_rate,
        channels=1
    )
    sound.export(wav_file, format='wav')


if __name__ == '__main__':
    input_file_path = "../test.wav"
    d = wav_to_array(input_file_path)
    print(str(d.shape))
    array_to_wav(d, "../machin.wav")
