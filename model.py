import yfinance as yf
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# if you want to see the data
# ticker_symbol = "AAPL"
# ticker = yf.Ticker(ticker_symbol)
# df = ticker.history(period="max")
# print(df.tail())

# Model Class
class Model(nn.Module):
    # Input Layers
    # Open, High, Low, Close, Volume x 7 days = 35 input neurons
    # The system tries to predict the close on the next day, so 1 output neuron

    def __init__(self, in_features=35, h1=8, h2=8, h3=8, out_features=1):
        super()
        self.fc1 = nn.Linear(in_features, h1)  # connect layers
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, out_features)

    def forward(self, x):
        x = F.relu(x)  # rectified linear unit, every x<0 is 0, else value stays the same
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x
