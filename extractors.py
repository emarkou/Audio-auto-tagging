#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import librosa
import numpy as np
import os

def extract_feature(file_name):
    '''
    extract_feature returns the features as specified by melspectrogram for each sound file
    '''
    Y, sample_rate =librosa.load(file_name,sr=2108)
    #features = librosa.feature.mfcc(y=Y, sr=sample_rate, n_mfcc=120)
    features = librosa.feature.melspectrogram(y=Y, sr=sample_rate, n_mels=120)
    return features

def generate_spectrogram(directory_files):
    filenames = os.listdir(directory_files)
    if not os.path.isdir(os.join.path(directory_files, "wav_to_npy")):
        os.mkdir(os.join.path(directory_files, "wav_to_npy"))
    for wav_file in filenames:
        npy = extract_feature(os.path.abspath(wav_file))
        dest = os.join.path(directory_files, "wav_to_npy", str(os.path.splitext(wav_file)[0]))
        np.save(dest, npy)
