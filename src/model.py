import torch
import torch.nn as nn

class AirQualityLSTM(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, num_layers=2, dropout=0.2, output_dim=1):
        super(AirQualityLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x is (batch_size, sequence_length, input_dim)
        out, _ = self.lstm(x)
        # We only need the output from the last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class AirQualityGRU(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, num_layers=2, dropout=0.2, output_dim=1):
        super(AirQualityGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x is (batch_size, sequence_length, input_dim)
        out, _ = self.gru(x)
        # We only need the output from the last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out
