import os

from dumb_audio_restoration.audio_degrader import degrade_directory
from dumb_audio_restoration.audio_splitter import split_directory

if not os.path.exists("../prepared_targets"):
    os.makedirs("../prepared_targets")
if not os.path.exists("../prepared_inputs"):
    os.makedirs("../prepared_inputs")

print("splitting training files to same length files")
split_directory("../lossless_source_for_training", "../prepared_targets")

print("degrading files")
degrade_directory("../prepared_targets", "../prepared_inputs")
