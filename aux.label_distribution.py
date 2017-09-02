#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

# Set directory path where downloaded csv with labels is stored
directory_labels = <file directory with file that holds labels>

###############################################################################
#                            Part 1: Label reduction
###############################################################################
# Load the annotations file 
#annotations_final.csv contains the initial labels (188) 
#annotations_final_new.csv contains the remaining labels (50)  
annotations_original = pd.read_csv(directory_labels + 'annotations_final.csv', sep="\t")
annotations_reducted = pd.read_csv(directory_labels + 'annotations_final_new.csv', sep=",")

# check size of annotations
annotations_original.info()
annotations_reducted.info()
# ensure correct loading
annotations_original.head(5)
annotations_reducted.head(5)
# initial number of columns: 190
annotations_original.columns 
annotations_reducted.columns 

# drop clip_id and mp3_path to proceeclip_filed with distribution counts
annotations_original = annotations_original[annotations_original.columns.difference(['clip_id', 'mp3_path'])]
annotations_original.info()

annotations_reducted = annotations_reducted[annotations_reducted.columns.difference(['Unnamed: 0', 'mp3_path', 'features_id'])]


# tags' distribution
orignal_tags_distro = annotations_original.sum(axis=0)
original_song_distro = annotations_original.sum(axis=1).value_counts()

reducted_tags_distro = annotations_reducted.sum(axis=0)
reducted_song_distro = annotations_reducted.sum(axis=1).value_counts()

#orignal_tags_distro.sort_values().plot(kind ='barh', figsize=(20,20))
original_song_distro.sort_index(axis = 0).plot(kind ='bar', figsize=(15,15))

reducted_tags_distro.sort_values().plot(kind ='barh', figsize=(20,20))
reducted_song_distro.sort_index(axis = 0).plot(kind ='bar', figsize=(15,15))

orignal_tags_distro.to_csv(directory_labels + 'orignal_tags_distro.csv', sep=',')


