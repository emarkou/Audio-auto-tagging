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


# Define directory where sound files are stored within '/wav_files_TOTAL/' folder
root = <directory with wav files>

os.chdir(root + '/wav_files_TOTAL/')
filenames = os.listdir('.')
print(len(filenames))

for wav_file in filenames:
    
    print(wav_file)
    npy = extract_feature(os.path.abspath(wav_file))
    
    dest = root + '/npy_files_TOTAL_train/' + str(os.path.splitext(wav_file)[0])
    print(dest)

    np.save(dest, npy)
