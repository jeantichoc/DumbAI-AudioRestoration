import os

from open_picky_ears.degrade_audio import degrade_directory
from open_picky_ears.split_audio import split_directory

os.makedirs("../prepared_target")
os.makedirs("../prepared_input")

print("splitting training files to same length files")
split_directory("../audio_target", "../prepared_target")

print("degrading files")
degrade_directory("../prepared_target", "../prepared_input")
