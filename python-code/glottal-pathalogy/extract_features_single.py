
from __future__ import print_function
import numpy as np
import scipy
import matplotlib.pyplot as plt

import librosa
import librosa.display
import os
from pathlib import Path
import filehelper


f="../audio/1205-a_h.wav"
print("Printing MelSpectogram for audio ---------------")
print(f)
fileName=Path(f).name
className=Path(f).parent.name
classesIndexMapRev=filehelper.read_object("../audio/labels_index.bin")
print(classesIndexMapRev)
y, sr = librosa.load(f)
# Set the hop length; at 22050 Hz, 512 samples ~= 23ms
hop_length = 512

# Separate harmonics and percussives into two waveforms
y_harmonic, y_percussive = librosa.effects.hpss(y)

# Beat track on the percussive signal
tempo, beat_frames = librosa.beat.beat_track(y=y_percussive,
											 sr=sr)

mfcc=librosa.feature.melspectrogram(y,sr=sr,n_mels=13)
#print(str(len(mfcc))+" "+str(len(mfcc[0])))
#print(mfcc)
mfcc_delta = librosa.feature.delta(mfcc)
mean_mfcc=mfcc.mean(axis=1)
#print(mean_mfcc)
a_str = ','.join(str(x) for x in mean_mfcc)  # '0,3,5'
print(a_str)