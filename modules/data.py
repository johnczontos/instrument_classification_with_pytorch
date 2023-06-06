import os
import torch
import torchaudio.transforms as T
import pandas as pd
from torch.utils.data import Dataset
import torchaudio

class AudioClassificationDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.class_names = self.data.columns[1:]
        self.num_classes = len(self.class_names)
        _, _, self.sample_rate = self.__getitem__(0)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_file = os.path.join(self.root_dir, self.data['filename'][idx])
        waveform, sample_rate = torchaudio.load(audio_file)

        label = self.data.iloc[idx, 1:].values.astype(float)  # Assumes one-hot encoding starts from column index 1
        label = torch.from_numpy(label).float()

        if self.transform is not None:
            waveform = self.transform(waveform)

        return waveform, label, sample_rate

if __name__=="__main__":
    root_dir = '/nfs/guille/eecs_research/soundbendor/zontosj/instrument_classification_with_pytorch/data/in'
    csv_file = '/nfs/guille/eecs_research/soundbendor/zontosj/instrument_classification_with_pytorch/data/dataset.csv'


    dataset = AudioClassificationDataset(root_dir, csv_file)
    print("len(dataset):", len(dataset))  # Prints the number of samples in the dataset

    waveform, label, sample_rate = dataset[0]  # Access the first sample
    print("waveform.shape:", waveform.shape)  # Prints the shape of the waveform tensor
    print("label:", label)  # Prints the label of the sample
    print("dataset.num_classes:", dataset.num_classes)
    print("sample_rate:", sample_rate)

# orig_freq=44100
# target_freq=16000
# sample_rate=16000
# mean=[0.5]
# std=[0.5]
# 
# transformation_pipeline = T.Compose([
#             T.Resample(orig_freq, target_freq),
#             T.MelSpectrogram(sample_rate),
#             T.Normalize(mean, std)
#         ])