import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, inputL=29, hiddenL1=150, hiddenL2=150, hiddenL3=150, outputL=1):
        super(LSTMModel, self).__init__()
        # LSTM-Schicht
        self.lstm = nn.LSTM(input_size=inputL, hidden_size=hiddenL1, batch_first=True)
        # Fully connected layers nach dem LSTM
        self.fullyConnected2 = nn.Linear(hiddenL1, hiddenL2)
        self.fullyConnected3 = nn.Linear(hiddenL2, hiddenL3)
        self.fullyConnected4 = nn.Linear(hiddenL3, outputL)  # Ausgabe ist 1, z.B. die Vorhersage für Tag 30

    def forward(self, x):
        # x ist der Eingabetensor der Form [Batch, Sequence Length, Features]
        lstm_out, (hidden, cell) = self.lstm(x)
        # Wir nehmen den letzten Hidden State
        x = hidden[-1]
        x = F.relu(self.fullyConnected2(x))
        x = F.relu(self.fullyConnected3(x))
        x = self.fullyConnected4(x)  # Endgültige Ausgabe

        return x
