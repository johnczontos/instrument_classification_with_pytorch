"""
Project: Instrument Classification
File: run_model.py
Authors: John Zontos
Date: June 5, 2023

Description:
Demo script.

Author: John Zontos

This script serves as the main entry point for the Conditional Musical Instrument Classification System.
"""
import torch
import torch.nn as nn

from modules.audio_transformer import AudioTransformer
from modules.audio_lstm import AudioLSTM
from modules.data import AudioClassificationDataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Instantiate the dataset
root_dir = '/nfs/guille/eecs_research/soundbendor/zontosj/instrument_classification_with_pytorch/data/in'
csv_file = '/nfs/guille/eecs_research/soundbendor/zontosj/instrument_classification_with_pytorch/data/dataset.csv'
dataset = AudioClassificationDataset(root_dir, csv_file)

# Set hyperparameters
input_size = 16000  # Specify the input size of the AudioLSTM model
hidden_size = 128
num_layers = 1
num_classes = dataset.num_classes
bidirectional = False

# Set training parameters
batch_size = 4
num_epochs = 10
learning_rate = 0.01

# Create a data loader for batching
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Instantiate the model
model = AudioLSTM(input_size, hidden_size, num_layers, num_classes, bidirectional)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
total_steps = len(data_loader)
for epoch in range(num_epochs):
    for i, (waveform, labels, _) in enumerate(data_loader):
        # Forward pass
        outputs = model(waveform)

        # Compute the loss
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print training progress
        if (i+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item()}")

# Save the trained model if desired
torch.save(model.state_dict(), 'audio_lstm_model.pth')