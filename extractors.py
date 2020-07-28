#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import librosa
import numpy as np
import os
from pathlib import Path


def extract_feature(file_name):
    '''
    extract_feature returns the features as specified by melspectrogram for each sound file
    '''
    Y, sample_rate =librosa.load(file_name,sr=2108)
    #features = librosa.feature.mfcc(y=Y, sr=sample_rate, n_mfcc=120)
    features = librosa.feature.melspectrogram(y=Y, sr=sample_rate, n_mels=120)
    return features

def generate_spectrogram(directory_files):
    spectrogram_save_path = os.path.join(Path(directory_files).parent, 'wav_to_npy')

    filenames = os.listdir(directory_files)
    if not os.path.isdir(spectrogram_save_path):
        os.mkdir(spectrogram_save_path)
    for wav_file in filenames:
        # TODO: find why some files are not renamed
        if wav_file[0].isdigit():
            npy = extract_feature(os.path.join(directory_files, wav_file))
            dest = os.path.join(spectrogram_save_path, str(os.path.splitext(wav_file)[0]))
            np.save(dest, npy)
