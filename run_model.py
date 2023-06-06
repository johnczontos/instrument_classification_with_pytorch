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
import torch.optim as optim
from torch.utils.data import DataLoader

# from modules.audio_transformer import AudioTransformer
from modules.audio_lstm import AudioLSTM
from modules.data import AudioClassificationDataset
from modules.trainer import Trainer
from utils.metrics import calculate_metrics

from sklearn.model_selection import train_test_split
import tqdm as tq

import neptune

# Instantiate Neptune
run = neptune.init_run(
    project="Soundbendor/instrument-classification",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmZjY2NzQ5Yi0wZDVjLTRiODktOTNlNy0xODg3YTRkZTVmYTcifQ==",
)

# Instantiate the dataset
root_dir = '/nfs/guille/eecs_research/soundbendor/zontosj/instrument_classification_with_pytorch/data/mini_test/in'
csv_file = '/nfs/guille/eecs_research/soundbendor/zontosj/instrument_classification_with_pytorch/data/mini_test/dataset.csv'
dataset = AudioClassificationDataset(root_dir, csv_file)

# hyperparams
params = {
    "learning_rate": 1e-2,
    "batch_size": 4,
    "num_epochs": 10,
    "input_size": dataset.wav_length,
    "num_classes": dataset.num_classes,
    "val_split": 0.1,
    "test_split": 0.3,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "model_name": "audio_lstm",
    "hidden_size": 128,
    "num_layers": 1,
    "bidirectional": False,
}

# Set the proportions for train, validation, and test splits
train_split = 1 - (params["test_split"] + params["val_split"])

# Split the dataset into train and remaining data
train_data, remaining_data = train_test_split(dataset, test_size=1 - train_split, random_state=42)

# Split the remaining data into validation and test sets
val_data, test_data = train_test_split(remaining_data, test_size=params["test_split"] / (params["val_split"] + params["test_split"]), random_state=42)

# Create data loaders for training and validation
train_loader = DataLoader(train_data, batch_size=params["batch_size"], shuffle=True)
val_loader = DataLoader(val_data, batch_size=params["batch_size"])
test_loader = DataLoader(test_data, batch_size=params["batch_size"])

# Instantiate the model
model = AudioLSTM(params["input_size"], params["hidden_size"], params["num_layers"], params["num_classes"], params["bidirectional"])

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

# Create model trainer
trainer = Trainer(model, criterion, optimizer, params, run)

# Training loop
total_steps = len(train_loader)
best_val_loss = None
best_val_accuracy = None
for epoch in range(params["num_epochs"]):
    # training phase
    pbar = tq.tqdm(desc="Epoch {}".format(epoch+1), total=total_steps, unit="steps")
    avg_train_loss, train_accuracy = trainer.train(train_loader, pbar)
    pbar.close()
        
    # Validation phase
    avg_val_loss, val_accuracy = trainer.validate(val_loader)

    if not best_val_loss or avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_val_accuracy = val_accuracy
        # torch.save(model, 'audio_lstm_model.pt')

    # Print progress
    print('------------------ Epoch [{}/{}] ------------------\n\
          Training Loss: {:.4f} Training Accuracy: {:.2f}\n\
          Validation Loss: {:.4f} Validation Accuracy: {:.2f}'.format(epoch + 1, params["num_epochs"], avg_train_loss, train_accuracy, avg_val_loss, val_accuracy))
    
# Evaluation
predicted, scores, labels = trainer.evaluate(test_loader)
results = calculate_metrics(predicted, scores, labels)

print('--------------------- Results ----------------------')
print('best_val_loss: {:.4f} best_val_accuracy: {:.2f}'.format(best_val_loss, best_val_accuracy))
print('Test Accuracy: {:.4f}'.format(results['accuracy']))
print('f1: {:.2f}'.format(results['f1']))
print('precision: {:.2f}'.format(results['precision']))
print('recall: {:.2f}'.format(results['recall']))
print('mcc: {:.2f}'.format(results['mcc']))
print('auc_roc: {:.2f}'.format(results['auc_roc']))
print('----------------------------------------------------')
run['results'] = results
run.stop()