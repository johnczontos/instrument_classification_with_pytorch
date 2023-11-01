import torch
import torch.nn as nn

class AudioLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, bidirectional=False, dropout=0.5):
        super(AudioLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=bidirectional, dropout=dropout)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)

    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size, device=x.device)

        # Forward propagate the LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])

        return out


if __name__=="__main__":
    input_size = 10
    hidden_size = 128
    num_layers = 2
    num_classes = 5

    model = AudioLSTM(input_size, hidden_size, num_layers, num_classes, bidirectional=True)
    input_tensor = torch.randn(32, 100, input_size)  # Batch size of 32, sequence length of 100
    output = model(input_tensor)
    print(output.shape)