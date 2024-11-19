from datetime import datetime
from data import data
import model as model_class
import torch
import visualize as v
import pandas as pd
from src.data.data import FORECASTHORIZON

# training setup
torch.manual_seed(41)
f_input=29
model = model_class.Model(inputL=f_input, hiddenL1=150, hiddenL2=150, hiddenL3=150, outputL=FORECASTHORIZON)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# get the data and scaler
x_train, x_test, y_train, y_test, rest_scaler, price_scaler = data.prepare_training_data(data.t_combined)

print("x_train:", x_train.shape)
print("x_test:", x_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
print("\n")

# Training loop
model.train()
epochs = 10000
losses = []
test_losses = []
prediction = []

for i in range(x_train.size(1)):
    # training
    y_pred = model(x_train[:,i,:])
    print (f"y_pred: {y_pred.shape}")
    print(f"y_train: {y_train.shape}")
    loss = criterion(y_pred, y_train) # y_triang = 29
    losses.append(loss.item())

    # validation
    model.eval()
    with torch.no_grad():
        y_pred_test = model(x_test)
        test_loss = criterion(y_pred_test, y_test)
        test_losses.append(test_loss.item())
    model.train()

 #   if i % 100 == 0:
 #       print(f"Epoch: {i}")
 #       print(f"Training loss: {loss.item():.4f}")
 #       print(f"Test loss: {test_loss.item():.4f}")
 #       print("\n")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    prediction = y_pred_test

# plot
v.plot_losses(losses, test_losses)
d_time_series = pd.read_excel('../data_xlsx/d_timeseries.xlsx', index_col=0)
d_time_series['date'] = d_time_series['date'].apply(lambda x: datetime.fromtimestamp(x).date()) # convert from unix timestamp to datetime
date=d_time_series['date']
date = date[:len(prediction)]


print(f"länge ytest: {len(y_test)}")
print(f"länge ypred: {len(prediction)}")
print(f"länge ypred: {len(date)}")
print(prediction.shape)
v.plot_stocks(date, y_test, prediction, scaler=price_scaler)
