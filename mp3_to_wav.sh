#!/bin/bash
#With below command we convert mp3 files to wav files
#and remove existing mp3 files from directory

cd $1
for file in *.mp3; do
       ffmpeg -i "$file" -acodec pcm_s16le -ac 1 -ar 44100 "${file%.mp3}".wav
done
rm *.mp3
