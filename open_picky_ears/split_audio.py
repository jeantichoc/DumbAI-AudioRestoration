import sys

from pydub import AudioSegment
import os


def split_audio_file(input_file_path, output_dir_path, segment_length_ms=20000):
    segments = []

    # load the audio file
    audio = AudioSegment.from_file(input_file_path)

    # get the total length of the audio file in milliseconds
    audio_length_ms = len(audio)

    # calculate the number of segments required to split the audio file
    num_segments = audio_length_ms // segment_length_ms

    # loop through each segment and save it as a separate WAV file
    for i in range(num_segments):
        # calculate the start and end time of the segment in milliseconds
        start_time = i * segment_length_ms
        end_time = (i + 1) * segment_length_ms

        # extract the segment from the audio file
        segment = audio[start_time:end_time]

        current_segment_length_ms = len(segment)
        if current_segment_length_ms < segment_length_ms:
            continue

        segments.append(segment)

        # construct the output file name
        output_file_name = os.path.splitext(os.path.basename(input_file_path))[0] + f"_segment_{i + 1}.wav"
        output_file_path = os.path.join(output_dir_path, output_file_name)

        # save the segment as a WAV file
        segment.export(output_file_path, format="wav")
    return segments


if __name__ == '__main__':
    input_file_path = sys.argv[1]
    # get the directory path and file name of the input file
    input_dir_path, input_file_name = os.path.split(input_file_path)
    output_dir_path = input_dir_path
    # split the audio file into segments in the output directory
    split_audio_file(input_file_path, output_dir_path)
