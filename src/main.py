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
model = model_class.LSTMModel(inputL=f_input, hiddenL1=150, hiddenL2=150, hiddenL3=150, outputL=1)
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
#epochs = x_train.size(1)
epochs = 100
losses = []
test_losses = []
prediction = []

for i in range(epochs):
    # training
    y_pred = model(x_train[:,:,:])

    loss = criterion(y_pred, y_train)
    losses.append(loss.item())
    print(i)
    # validation
    model.eval()
    with torch.no_grad():
        y_pred_test = model(x_test)
        test_loss = criterion(y_pred_test, y_test)
        test_losses.append(test_loss.item())
    model.train()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    prediction = y_pred_test

# save data
np_y_pred = y_pred.numpy()
df = pd.DataFrame(np_y_pred)
df.to_csv("../model_output/y_pred.csv", index=False)

np_losses = losses.numpy()
df = pd.DataFrame(np_losses)
df.to_csv("../model_output/losses.csv", index=False)

np_test_losses = test_losses.numpy()
df = pd.DataFrame(np_test_losses)
df.to_csv("../model_output/test_losses.csv", index=False)

# plot
v.plot_losses(losses, test_losses)
d_time_series = pd.read_excel('../data_xlsx/d_timeseries.xlsx', index_col=0)
d_time_series['date'] = d_time_series['date'].apply(lambda x: datetime.fromtimestamp(x).date()) # convert from unix timestamp to datetime
date = d_time_series['date']
date = date[:len(prediction)]


print(prediction.shape)
date = date[-len(y_test):]  # Schneide die Daten auf die Länge von y_test zu
v.plot_stocks(date, y_test, prediction, scaler=price_scaler)
