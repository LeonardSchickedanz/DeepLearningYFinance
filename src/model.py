import torch.nn as nn
import torch.nn.functional as F

# Model Class
class Model(nn.Module):
    # Input Layers
    # Open, High, Low, Close, Volume x 7 days = 35 input neurons
    # The system tries to predict the close on the next day, so 1 output neuron

    def __init__(self, inputL=30, hiddenL1=8, hiddenL2=8, hiddenL3=8, outputL=1):
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

