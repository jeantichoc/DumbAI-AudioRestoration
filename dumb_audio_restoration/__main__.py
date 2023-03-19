from dumb_audio_restoration.apply_model_on_one_file import apply_model_on_file
from dumb_audio_restoration.prepare import prepare_files_for_training
from dumb_audio_restoration.train_model import train_model

if __name__ == '__main__':
    prepare_files_for_training()

    trained_model = train_model()
    trained_model.save("../trained_model")

    apply_model_on_file("../tests/test.wav", "../tests/restored.wav", "../trained_model")
