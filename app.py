from label_reduction_and_files_renaming import perform_label_reduction, file_renaming
from config import directory_files, directory_labels
import os

annotations, clip_file = perform_label_reduction(directory_labels)
file_renaming(annotations, directory_files)

print("finished!")

