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

        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        self.fc = nn.Linear(hidden_size, 1)


    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)

        attention_scores = self.attention_layer(lstm_out)
        attention_weights = torch.softmax(attention_scores, dim=1)

        weighted_out = lstm_out * attention_weights
        context_vector = torch.sum(weighted_out, dim=1)

        prediction = self.fc(context_vector)

        if return_attention:
            return prediction, attention_weights
        return prediction