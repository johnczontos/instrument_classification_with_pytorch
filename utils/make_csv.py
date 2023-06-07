import os
import csv
import glob

# Define the directory path
data_path = "/nfs/guille/eecs_research/soundbendor/zontosj/instrument_classification_with_pytorch/data/piano_guitar_violin/in"

# Get the list of directories
dirs = [d.name for d in os.scandir(data_path) if d.is_dir()]

header = ["Filename"] + dirs
data = []

# Iterate over each directory
for i, directory in enumerate(dirs):
    # Get all the audio files in the directory
    audio_files = glob.glob(os.path.join(data_path, directory, "*.wav"))

    # Generate the one-hot encoded label code
    code = [0] * len(dirs)
    code[i] = 1

    # Append the file name and label code to the data list
    for audio_file in audio_files:
        data.append([os.path.basename(audio_file)] + code)

# Write the data to the CSV file
with open("dataset.csv", "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(header)
    writer.writerows(data)




# data directory structure example
# __data
# |____...
# |____clean
# | |____piano
# | | |____Piano.mf.Eb3.wav
# | | |____Piano.mf.Gb5.wav
# | | |____Piano.ff.Db5.wav
# | | |____...
# | |____guitar
# | | |____Guitar.mf.Eb3.wav
# | | |____Guitar.mf.Gb5.wav
# | | |____Guitar.ff.Db5.wav
# | | |____...
# | |____...
