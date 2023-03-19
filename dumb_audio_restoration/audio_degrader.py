import os
from pydub import AudioSegment


def degrade_audio(input_file_path, output_dir_path):
    # load the audio file
    audio = AudioSegment.from_file(input_file_path)

    # construct the output file name
    input_file_name = os.path.basename(input_file_path)
    output_file_name = os.path.splitext(input_file_name)[0] + "_degraded.wav"
    output_file_path = os.path.join(output_dir_path, output_file_name)
    print("degrading ... " + output_file_path)

    # export the audio as a degraded MP3 file
    audio.export(output_file_path, format="mp3", bitrate="64k")

    # load the degraded MP3 file
    degraded_audio = AudioSegment.from_file(output_file_path, format="mp3")

    # export the degraded audio as a WAV file
    degraded_audio.export(output_file_path, format="wav")


def degrade_directory(input_dir_path, output_dir_path):
    # loop through each file in the input directory tree
    for root, dirs, files in os.walk(input_dir_path):
        for file in files:
            # check if the file is an audio file
            if file.endswith(".wav") or file.endswith(".flac"):
                # construct the input and output file paths
                input_file_path = os.path.join(root, file)
                output_file_dir = os.path.join(output_dir_path, os.path.relpath(root, input_dir_path))

                # create the output directory if it doesn't exist
                if not os.path.exists(output_file_dir):
                    os.makedirs(output_file_dir)

                # degrade the audio and save the degraded audio file
                degrade_audio(input_file_path, output_file_dir)
