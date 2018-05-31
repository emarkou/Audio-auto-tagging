#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import os
import shutil

# Set directory path where downloaded csv with labels is stored
directory_labels = <path to file with labels>

# Set directory path where downloaded mp3 files are stored in a COMMON FOLDER
# i.e. all subfolders have been removed using the following bash script:
# find . -mindepth 2 -type f -print -exec mv {} . \;
directory_files = <path to mp3 files>



###############################################################################
#                            Part 1: Label reduction
###############################################################################

# Load the annotations file (prior to execution check correctness of delimiter)
annotations = pd.read_csv(directory_labels + 'annotations_final.csv', sep="\t")

# check size of annotations
annotations.info()
# ensure correct loading
annotations.head(5)
# initial number of columns: 190
annotations.columns 

# merge synonym tags
synonyms = [['beat', 'beats'],
            ['chant', 'chanting'],
            ['choir', 'choral'],
            ['classical', 'clasical', 'classic'],
            ['drum', 'drums'],
            ['electro', 'electronic', 'electronica', 'electric'],
            ['fast', 'fast beat', 'quick'],
            ['female', 'female singer', 'female singing', 'female vocals', 'female vocal', 'female voice', 'woman', 'woman singing', 'women'],
            ['flute', 'flutes'],
            ['guitar', 'guitars'],
            ['hard', 'hard rock'],
            ['harpsichord', 'harpsicord'],
            ['heavy', 'heavy metal', 'metal'],
            ['horn', 'horns'],
            ['india', 'indian'],
            ['jazz', 'jazzy'],
            ['male', 'male singer', 'male vocal', 'male vocals', 'male voice', 'man', 'man singing', 'men'],
            ['no beat', 'no drums'],
            ['no singer', 'no singing', 'no vocal','no vocals', 'no voice', 'no voices', 'instrumental'],
            ['opera', 'operatic'],
            ['orchestra', 'orchestral'],
            ['quiet', 'silence'],
            ['singer', 'singing'],
            ['space', 'spacey'],
            ['string', 'strings'],
            ['synth', 'synthesizer'],
            ['violin', 'violins'],
            ['vocal', 'vocals', 'voice', 'voices'],
            ['strange', 'weird']]

# replace all synonym words with the first in the list constructed above
for synonym_list in synonyms:
    annotations[synonym_list[0]] = annotations[synonym_list].max(axis=1)
    annotations.drop(synonym_list[1:], axis=1, inplace=True)

# check results
# previously: Columns: 190 entries, clip_id to mp3_path &  memory usage: 37.5+ MB
# now: Columns: 136 entries, clip_id to mp3_path & memory usage: 26.8+ MB
annotations.info()

# keep clip_id and mp3_path in a different matrix to bind later
annotations_ids = annotations[['clip_id', 'mp3_path']]

# drop clip_id and mp3_path to proceed with distribution counts
annotations = annotations[annotations.columns.difference(['clip_id', 'mp3_path'])]

# tags' distribution
tags_distro = annotations.sum(axis=0)
# sort tags according to their occurencies 
tags_distro.sort_values(axis=0, inplace=True, ascending = False)

#find 50 top tags
topindex, topvalues = list(tags_distro.index[:50]), tags_distro.values[:50]

# list of columns to remove from annotation file (not common labels)
rem_cols_index =list(tags_distro.index[50:])
#check how many columns we are about to remove
len(rem_cols_index) 

# keep only the 50 most common tags
annotations.drop(rem_cols_index, axis=1, inplace=True)

# bind back clip_id and mp3_path
annotations['clip_id'] = annotations_ids['clip_id']
annotations['mp3_path'] = annotations_ids['mp3_path']

# ensure correct binding
annotations.head(5)

# check results
# previously: Columns: 136 entries, clip_id to mp3_path & memory usage: 26.8+ MB
# now: Columns: 52 entries, clip_id to mp3_path &  memory usage: 10.1 MB
annotations.info()

# Create concatenation of remaining features for each file
annotations_f = annotations[annotations.columns.difference(['clip_id', 'mp3_path'])]
annotations_f['features_id'] = annotations_f.apply(lambda x: ''.join(x.astype(str)), axis=1)
annotations['features_id']=annotations_f['features_id'].astype(str)+'_'+annotations['clip_id'].astype(str)
annotations.drop('clip_id', axis=1, inplace=True)

# extract clip_id and wav file path : 25863 rows x 2 columns
clip_file = annotations[['features_id', 'mp3_path']]
clip_file.head(5)

# turn previous dataframe into a matrix
features_id, mp3_path = annotations[["features_id", "mp3_path"]].as_matrix()[:,0], annotations[["features_id", "mp3_path"]].as_matrix()[:,1]

# some more checks
features_id[:5]
mp3_path[:5]

# save annotations & clip_file for reference
annotations.to_csv(directory_labels + 'annotations_final_new.csv', sep=',')
clip_file.to_csv(directory_labels + 'clip_file.csv', sep=',')



###############################################################################
#                Part 2: File renaming according to features_id
###############################################################################

# Rename all the mp3 files to their features_id
# and save into a new folder named 'features_id_mp3'

# Create destination directory where renamed mp3's will be stored
os.mkdir(directory_files + "features_id_mp3/")

# Iterate over the mp3 files, rename them to their features_id and save to destination
for id in range(25863):
    if features_id[id][:50] != '0'*50 :
        src = directory_files + mp3_path[id][2:]
        dest = directory_files + "features_id_mp3/" + str(features_id[id]) + ".mp3"
        shutil.copy2(src,dest)

