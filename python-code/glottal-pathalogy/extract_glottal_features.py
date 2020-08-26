
from __future__ import print_function
import numpy as np
import scipy
import matplotlib.pyplot as plt

import librosa
import librosa.display
import os
from pathlib import Path
import filehelper
#######################################################################
# We'll use a track that has harmonic, melodic, and percussive elements
TRAINING_DIR= "D:\\work\\project-2019\\Music-GlottalPathalogyDetection\\GlottalPathalogyDetection\\dataset\\patient-vocal-dataset\\"
classes=[]
classesIndex=0;
classesIndexMap={}
classesIndexMapRev={}
for f in os.listdir(TRAINING_DIR):
    print(f)
    classes.append(f)
    classesIndexMap[f]=classesIndex
    classesIndexMapRev[classesIndex]=f
    classesIndex=classesIndex+1
filehelper.save_object(classesIndexMap,"../audio/labels.bin")
filehelper.save_object(classesIndexMapRev,"../audio/labels_index.bin")
x = [os.path.join(r, file) for r, d, f in os.walk(TRAINING_DIR) for file in f]
x.sort(key=os.path.getmtime)

# print(x)
# x.sort(key=lambda x: os.path.getmtime(x))
for f in x:
    #print(f)
    fileName=Path(f).name
    className=Path(f).parent.name
    classesIndex=classesIndexMap[className]
    try:
        y, sr = librosa.load(f)
        # Set the hop length; at 22050 Hz, 512 samples ~= 23ms
        hop_length = 512

        # Separate harmonics and percussives into two waveforms
        y_harmonic, y_percussive = librosa.effects.hpss(y)

        # Beat track on the percussive signal
        tempo, beat_frames = librosa.beat.beat_track(y=y_percussive,
                                                     sr=sr)

        mfcc = librosa.feature.melspectrogram(y, sr=sr, n_mels=13)
        # print(str(len(mfcc))+" "+str(len(mfcc[0])))
        # print(mfcc)
        mfcc_delta = librosa.feature.delta(mfcc)
        mean_mfcc = mfcc.mean(axis=1)
        # print(mean_mfcc)
        a_str = ','.join(str(x) for x in mean_mfcc)  # '0,3,5'
        # s=np.array2string(mean_mfcc, precision=4, separator=',',suppress_small=False)
        # s=s+","+str(classesIndex)
        # s=s.replace("[","")
        # s = s.replace("]", "")
        print(a_str + "," + str(classesIndex))
    except:
        print("An exception occurred")


