import torch.nn as nn
import torch.nn.functional as F
from src.data.data import FORECASTHORIZON

# Model Class
class Model(nn.Module):
    # Input Layers

    def __init__(self, inputL=29, hiddenL1=150, hiddenL2=150, hiddenL3=150, outputL=FORECASTHORIZON):
        super().__init__()
        self.fullyConnected1 = nn.Linear(inputL, hiddenL1)  # connect layers
        self.fullyConnected2 = nn.Linear(hiddenL1, hiddenL2)
        self.fullyConnected3 = nn.Linear(hiddenL2, hiddenL3)
        self.fullyConnected4 = nn.Linear(hiddenL3, outputL)

    def forward(self, x):
        x = F.relu(self.fullyConnected1(x))  # rectified linear unit, every x<0 is 0, else value stays the same
        x = F.relu(self.fullyConnected2(x))
        x = F.relu(self.fullyConnected3(x))
        x = self.fullyConnected4(x)

        return x

