import torch
from neptune_pytorch import NeptuneLogger

class Trainer:
    def __init__(self, model, criterion, optimizer, run, device):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        # Neptune pytorch logger
        self.npt_logger = NeptuneLogger(
            run=run,
            model=model,
            log_model_diagram=True,
            log_gradients=True,
            log_parameters=True,
            log_freq=30,
            base_namespace="logging"
        )

        self.run = run


    def train(self, train_loader, pbar=None):
        self.model.train()
        epoch_loss = 0.0
        total = 0
        correct = 0
        for i, (waveform, labels) in enumerate(train_loader):
            # Move data to device
            waveform, labels = waveform.to(self.device), labels.to(self.device)

            # Forward pass
            outputs = self.model(waveform)

            # Compute the loss
            batch_loss = self.criterion(outputs, labels)
            epoch_loss += batch_loss.item()

            # Backward and optimize
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
            
            # Book keeping
            predicted = torch.argmax(outputs, dim=1)
            labels = torch.argmax(labels, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Log after every 10 steps
            # if i % 10 == 0:
            #     self.run[self.npt_logger.base_namespace]["train/batch/loss"].append(batch_loss.item())
            #     self.run[self.npt_logger.base_namespace]["train/batch/acc"].append(correct / total)

            if pbar:
                pbar.update(1)
        accuracy = correct / total
        avg_loss = epoch_loss / len(train_loader)
        # self.npt_logger.save_model("model")
        return avg_loss, accuracy

    def validate(self, val_loader):
        self.model.eval()
        epoch_loss = 0.0
        total = 0
        correct = 0
        with torch.no_grad():
            for i, (waveform, labels) in enumerate(val_loader):
                # Move data to device
                waveform, labels = waveform.to(self.device), labels.to(self.device)
            
                # Forward pass
                outputs = self.model.forward(waveform)

                # Compute the loss
                batch_loss = self.criterion(outputs, labels)
                epoch_loss += batch_loss.item()

                # Book keeping
                predicted = torch.argmax(outputs, dim=1)
                labels = torch.argmax(labels, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Log after every 10 steps
                # if i % 10 == 0:
                #     self.run[self.npt_logger.base_namespace]["validation/batch/loss"].append(batch_loss.item())
                #     self.run[self.npt_logger.base_namespace]["validation/batch/acc"].append(correct / total)

            accuracy = correct / total
            avg_loss = epoch_loss / len(val_loader)
        return avg_loss, accuracy
    
    def evaluate(self, data_loader):
        self.model.eval()
        predicted_labels = []
        predicted_scores = []
        true_labels = []

        with torch.no_grad():
            for waveform, labels in data_loader:
                waveform, labels = waveform.to(self.device), labels.to(self.device)
                outputs = self.model(waveform)
                batch_predicted_scores = torch.softmax(outputs, dim=1)
                batch_predicted = torch.argmax(outputs, dim=1)
                labels = torch.argmax(labels, dim=1)
                true_labels.extend(labels.tolist())
                predicted_labels.extend(batch_predicted.tolist())
                predicted_scores.extend(batch_predicted_scores.tolist())

        return predicted_labels, predicted_scores, true_labels
