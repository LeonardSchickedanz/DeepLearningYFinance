import torch
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# Download historical data
data = yf.download("AAPL", period = "20y")

# Create input and output data
def create_dataset(data, input_size=30, output_size=7):
    X, Y = [], []
    for i in range(len(data) - input_size - output_size + 1):
        X.append(data[i:(i + input_size)])
        Y.append(data["Close"].values[i + input_size:i + input_size + output_size])
    return np.array(X), np.array(Y)

X, Y = create_dataset(data)

# Train Test Split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=41)

# reshape data for dimension of model (30 neurons in inputL)
Xtrain = torch.FloatTensor(Xtrain).reshape(-1, 30)
Xtest = torch.Floattensor(Xtest).reshape(-1, 30)

# convert Y to tensors
Ytrain = torch.FloatTensor(Ytrain)
Ytest = torch.FloatTensor(Ytest)

