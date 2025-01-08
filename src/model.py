import torch.nn as nn
import torch.nn.functional as F
import torch

class LSTMModel(nn.Module):
    def __init__(self, inputL=38, hiddenL1=150, hiddenL2=150, hiddenL3=150, outputL=1):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_size=inputL, hidden_size=hiddenL1, batch_first=True)

        self.fullyConnected2 = nn.Linear(hiddenL1, hiddenL2)
        self.fullyConnected3 = nn.Linear(hiddenL2, hiddenL3)
        self.fullyConnected4 = nn.Linear(hiddenL3, outputL)

    def forward(self, x, hidden_state = None, cell_state = None):

        if hidden_state is None or cell_state is None:
            hidden_state = torch.zeros(1, x.size(0), self.lstm.hidden_size).to(x.device)
            cell_state = torch.zeros(1, x.size(0), self.lstm.hidden_size).to(x.device)

        lstm_out, (hidden, cell) = self.lstm(x, (hidden_state, cell_state))

        # last hidden state
        x = hidden[-1]
        x = F.relu(self.fullyConnected2(x))
        x = F.relu(self.fullyConnected3(x))
        x = self.fullyConnected4(x)

        return x, (hidden, cell)
