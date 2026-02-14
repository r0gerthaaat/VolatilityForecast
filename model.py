import torch
import torch.nn as nn

class VolatilityLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.2):
        super(VolatilityLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)


    def forward(self, x: torch.tensor()) -> torch.tensor():
        lstm_out, _ = self.lstm(x)

        last_out = lstm_out[:, -1, :]

        prediction = self.fc(last_out)
        return prediction
