#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os

def perform_label_reduction(directory_labels):
    annotations = pd.read_csv(os.path.join(directory_labels, 'annotations_final.csv'), sep="\t")

    # Synonym tags to merge
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

    for synonym_list in synonyms:
        annotations[synonym_list[0]] = annotations[synonym_list].max(axis=1)
        annotations.drop(synonym_list[1:], axis=1, inplace=True)

    annotations_ids = annotations[['clip_id', 'mp3_path']]
    annotations = annotations[annotations.columns.difference(['clip_id', 'mp3_path'])]
    tags_distro = annotations.sum(axis=0)
    tags_distro.sort_values(axis=0, inplace=True, ascending = False)
    rem_cols_index = list(tags_distro.index[50:])

    # keep only the 50 most common tags
    annotations.drop(rem_cols_index, axis=1, inplace=True)
    annotations['clip_id'] = annotations_ids['clip_id']
    annotations['mp3_path'] = annotations_ids['mp3_path']
    annotations_f = annotations[annotations.columns.difference(['clip_id', 'mp3_path'])]
    annotations_f['features_id'] = annotations_f.apply(lambda x: ''.join(x.astype(str)), axis=1)
    annotations['features_id']=annotations_f['features_id'].astype(str)+'_'+annotations['clip_id'].astype(str)
    annotations.drop('clip_id', axis=1, inplace=True)
    clip_file = annotations[['features_id', 'mp3_path']]

    annotations.to_csv(directory_labels + 'annotations_final_new.csv', sep=',')
    clip_file.to_csv(directory_labels + 'clip_file.csv', sep=',')

    return annotations, clip_file

def file_renaming(annotations, directory_files):
    features_id = annotations["features_id"].tolist()
    mp3_path = annotations["mp3_path"].tolist()

    _, _, files = next(os.walk(directory_files))
    for id in range(len(files)):
        if features_id[id][:50] != '0'*50:
            src = os.path.join(directory_files + mp3_path[id][2:])
            dest = os.path.join(directory_files, str(features_id[id])+".mp3")
            os.rename(src, dest)

