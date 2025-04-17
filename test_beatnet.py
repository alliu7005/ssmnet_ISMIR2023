from BeatNet.BeatNet import BeatNet

import os
import librosa

SONG_FOLDER = "songs/song1/"


path = os.path.join(SONG_FOLDER, "song1" + ".mp3")

y, sr = librosa.load(path)

beat_tracker = BeatNet(1, mode="offline", inference_model="DBN", plot=['beat_particles'], thread=False)

print(beat_tracker.process(y))