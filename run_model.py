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

input_size = 10
hidden_size = 128
num_layers = 2
num_classes = 5

model = AudioLSTM(input_size, hidden_size, num_layers, num_classes, bidirectional=True)
input_tensor = torch.randn(32, 100, input_size)  # Batch size of 32, sequence length of 100
output = model(input_tensor)
print(output.shape)