import os
import shutil

source_dir = "/nfs/guille/eecs_research/soundbendor/datasets/RWC/MusicalInstrumentSoundDatabase"
target_dir = "/nfs/guille/eecs_research/soundbendor/zontosj/instrument_classification_with_pytorch/data/rwc_all"

for subdir in os.listdir(source_dir):
    if not subdir.startswith(".") and os.path.isdir(os.path.join(source_dir, subdir)):
        print(subdir)
        subdir_path = os.path.join(source_dir, subdir)
        subdir_prefix = subdir[:2]

        for file in os.listdir(subdir_path):
            print(file)
            if file.endswith(".wav") or file.endswith(".WAV"):
                file_path = os.path.join(subdir_path, file)
                target_subdir = os.path.join(target_dir, subdir_prefix)

                # Create the target subdirectory if it doesn't exist
                if not os.path.exists(target_subdir):
                    os.makedirs(target_subdir)

                # Copy or move the file to the target subdirectory
                shutil.copy2(file_path, target_subdir)  # Change to shutil.move if you want to move the files instead of copying
