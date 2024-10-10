import model
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import torch
import data
from sklearn.model_selection import train_test_split
print("test")
# Pick manual seed
torch.manual_seed(41)

# Create model
# model = model.Model()

# set criterion
criterion = torch.nn.CrossentropyLoss()

# choose adam optimizer, lr = learning rate (if error doesn't go down after a bunch of iterations lower lr)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # parameters are layers

# Train model
# Epochs (one run through all the training data)

Xtrain = data.Xtrain
Ytrain = data.Ytrain

epochs = 100
losses = []
for i in range(epochs):
    # go forward and get a prediction
    Ypred = model.forward(Xtrain)  # get predicted results

    # measure loss/error
    loss = criterion(Ypred, Ytrain)  # predicted values vs the Ytrain

    # keep track of losses
    losses.append(loss.detach().numpy())

    # print every 10 epoch
    if i % 10 == 0:
        print("Epoch: " + i + " loss: " + loss)

    # back propagation
    # back propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()