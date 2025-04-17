import librosa_test
from song_struct import Song_Struct
import csv
import numpy as np
import os
import librosa

folder_path = os.path.join(os.getcwd(), "songs")
song_folder = os.path.join(folder_path, "song6")

song_file = os.path.join(song_folder, "song6.mp3")
bass_file = os.path.join(song_folder,"bass.mp3")
drums_file = os.path.join(song_folder,"drums.mp3")
chords_file = os.path.join(song_folder,"other.mp3")
vocals_file = os.path.join(song_folder,"vocals.mp3")

y_orig, sr_orig = librosa.load(song_file)
        
y_bass, sr_bass = librosa.load(bass_file)
y_drums, sr_drums = librosa.load(drums_file)
y_other, sr_other = librosa.load(chords_file)
y_vocals, sr_vocals = librosa.load(vocals_file)

y_min = min(len(y_bass), len(y_drums), len(y_other), len(y_vocals)) - 1

y_other = y_other[0:y_min]
y_drums = y_drums[0:y_min]
y_bass = y_bass[0:y_min]
y_vocals = y_vocals[0:y_min]


song = Song_Struct(y_orig,sr_orig,y_vocals,y_drums,y_bass,y_other, "Counting Stars")

song.set_silence()
song.set_active()

song.to_string()
song.set_bounds()
song.visualize()
