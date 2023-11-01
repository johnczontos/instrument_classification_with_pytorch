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
from torch.utils.data import DataLoader, random_split

# from modules.audio_transformer import AudioTransformer
from modules.audio_lstm import AudioLSTM
from modules.data import AudioClassificationDataset
from modules.trainer import Trainer
from utils.metrics import calculate_metrics, print_results

# from sklearn.model_selection import train_test_split
import random
import tqdm as tq
import sys
import os

import neptune

print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(root_dir, csv_file):
    # Instantiate Neptune
    run = neptune.init_run(
        project="Soundbendor/instrument-classification",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmZjY2NzQ5Yi0wZDVjLTRiODktOTNlNy0xODg3YTRkZTVmYTcifQ==",
    )

    # Instantiate the dataset
    wav_length = 16000
    dataset = AudioClassificationDataset(root_dir, csv_file, 16000)
    print("INFO: Dataset loaded.")
    dataset.check_audio()
    dataset.info()
    # TODO: add logging instead of printing.

    # hyperparams
    params = {
        "learning_rate": 4e-4,
        "dropout": 0.5,
        "batch_size": 512,
        "num_epochs": 50,
        "input_size": dataset.wav_length,
        "num_classes": dataset.num_classes,
        "data_length": len(dataset),
        "val_split": 0.1,
        "test_split": 0.1,
        "device": device,
        "model_name": "audio_lstm",
        "hidden_size": 512,
        "num_layers": 4,
        "bidirectional": True,
    }

    # Set a random seed for reproducibility and shuffle dataset
    random.seed(42)

    # Set the proportions for train, validation, and test splits
    train_split = 1 - (params["test_split"] + params["val_split"])

    # Perform the splits
    train_data, val_data, test_data = random_split(dataset, [train_split, params["val_split"], params["test_split"]])

    # Create data loaders for training and validation
    train_loader = DataLoader(train_data, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=params["batch_size"])
    test_loader = DataLoader(test_data, batch_size=params["batch_size"])

    print("INFO: Dataloaders created.")

    # Instantiate the model
    model = AudioLSTM(params["input_size"], params["hidden_size"], params["num_layers"], params["num_classes"], params["bidirectional"], params["dropout"])
    print("INFO: Model created.")

    if torch.cuda.device_count() > 1:
        print("Multiple GPUs available, using: " + str(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

    # Create model trainer
    trainer = Trainer(model, criterion, optimizer, params, run, device)
    print("INFO: Trainer Created.")

    # Training loop
    total_steps = len(train_loader)
    best_val_accuracy = None
    for epoch in range(params["num_epochs"]):
        # training phase
        pbar = tq.tqdm(desc="Epoch {}".format(epoch+1), total=total_steps, unit="steps")
        avg_train_loss, train_accuracy = trainer.train(train_loader, pbar)
        pbar.close()
            
        # Validation phase
        avg_val_loss, val_accuracy = trainer.validate(val_loader)

        if not best_val_accuracy or val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            # torch.save(model, 'audio_lstm_model.pt')

        # Print progress
        print('------------------ Epoch [{}/{}] ------------------'.format(epoch + 1, params["num_epochs"]))
        print('Training Loss: {:.4f} Training Accuracy: {:.2f}'.format(avg_train_loss, train_accuracy))
        print('Validation Loss: {:.4f} Validation Accuracy: {:.2f}'.format(avg_val_loss, val_accuracy))

        run["logging/train/epoch/loss"].append(avg_train_loss)
        run["logging/train/epoch/acc"].append(train_accuracy)

        run["logging/validation/epoch/loss"].append(avg_val_loss)
        run["logging/validation/epoch/acc"].append(val_accuracy)

        
    # Evaluation
    print("INFO: Evaluation.")
    predicted, scores, labels = trainer.evaluate(test_loader)
    results = calculate_metrics(predicted, scores, labels)
    print_results(results, best_val_accuracy)
    run['results'] = results
    run.stop()


if __name__ == "__main__":
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 2:
        print("Error: Please provide the root directory as command-line argument.")
        print("Usage: python script.py <root_dir>")
        sys.exit(1)

    root_dir = sys.argv[1]
    csv_file = os.path.join(root_dir, "dataset.csv")

    # Call the main function with the provided paths
    main(root_dir, csv_file)