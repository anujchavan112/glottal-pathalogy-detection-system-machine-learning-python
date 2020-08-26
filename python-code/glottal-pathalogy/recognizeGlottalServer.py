import sys

sys.path.append('C:\\Python35\\Lib\site-packages')

import os
from glob import glob

import socket
import sys

import sys
from scipy.io import arff
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import filehelper
import librosa
from train_glottal_dataset import *

HOST = ''  # Symbolic name, meaning all available interfaces
PORT = 7813  # Arbitrary non-privileged port

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')
svclassifier = filehelper.read_object("../audio/svm.model")
y, sr = librosa.load("../audio/1205-a_h.wav")
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
a_str = ','.join(str(x) for x in mean_mfcc)
data_train = []
data_train.append(mean_mfcc)
data = svclassifier.predict(data_train)
print(data)

classesIndexMapRev = filehelper.read_object("../audio/labels_index.bin")
print(classesIndexMapRev)
# Bind socket to local host and port
try:
    s.bind((HOST, PORT))
except socket.error as msg:
    print('Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1])
    sys.exit()

print('Socket bind complete')

# Start listening on socket
s.listen(10)
print('Socket now listening')

# now keep talking with the client
while 1:
    # wait to accept a connection - blocking call
    conn, addr = s.accept()
    print('Connected with ' + addr[0] + ':' + str(addr[1]))
    try:
        data = conn.recv(1024).decode()
        print(data)
        audio_path = data.strip()
        print(audio_path == "1")
        if (audio_path == "1"):
            r = str(classesIndexMapRev)
            print(r)
            conn.send(r.encode())
            conn.close()
            continue
        elif (audio_path == "2"):
            s2=trainNetwork()
            conn.send(s2.encode())
            conn.close()
            continue
        else:
            exists = os.path.isfile(audio_path)
            print(exists)
            if (exists):
                filename = audio_path
                y, sr = librosa.load(filename)
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
                a_str = ','.join(str(x) for x in mean_mfcc)
                data_train = []
                data_train.append(mean_mfcc)
                data = svclassifier.predict(data_train)
                print(data)
                r = str(data[0]) + '#' + classesIndexMapRev[int(data[0])] + '#' + str(classesIndexMapRev) + "#" + a_str
                print(r)
                conn.send(r.encode())
    except Exception as e:
        print("An exception occurred")
        print(e.message)
    conn.close()
s.close()
