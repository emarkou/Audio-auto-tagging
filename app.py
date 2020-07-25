from converters import perform_label_reduction, file_renaming
from extractors import generate_spectrogram
from config import directory_files, directory_labels
from subprocess import call

annotations, clip_file = perform_label_reduction(directory_labels)
file_renaming(annotations, directory_files)

# Call bash script that converts mp3 to wav files
call(["./mp3_to_wav.sh", directory_files])
generate_spectrogram(directory_files)

print("finished!")

