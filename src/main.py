from datetime import datetime
import numpy as np
from data import data
import model as model_class
import torch
import visualize as v
import pandas as pd

torch.manual_seed(41)
model = model_class.LSTMModel(inputL = data.T_COMBINED.shape[1], hiddenL1=100, hiddenL2=100, hiddenL3=100, outputL=1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
x_train, x_test, y_train, y_test, rest_scaler, price_scaler = data.prepare_training_data(data.T_COMBINED)

def run():
    # get the data and scaler
    x_train, x_test, y_train, y_test, rest_scaler, price_scaler = data.prepare_training_data(data.T_COMBINED)

    print("x_train:", x_train.shape)
    print("x_test:", x_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)
    print("\n")

    # Training loop
    model.train()
    #epochs = x_train.size(1)
    epochs = 10
    losses = []
    test_losses = []
    prediction = []

    for epoch in range(epochs):
        # Initialisierung des Hidden- und Cell-States für Training
        hidden_state = torch.zeros(1, x_train.size(0), model.lstm.hidden_size).to(x_train.device)
        cell_state = torch.zeros(1, x_train.size(0), model.lstm.hidden_size).to(x_train.device)

        # training
        y_pred, (hidden_state, cell_state) = model(x_train[:,:,:], hidden_state, cell_state)

        loss = criterion(y_pred, y_train)
        losses.append(loss.item())
        print(epoch)
        # validation
        model.eval()
        with torch.no_grad():
            hidden_state_val = torch.zeros(1, x_test.size(0), model.lstm.hidden_size).to(x_test.device)
            cell_state_val = torch.zeros(1, x_test.size(0), model.lstm.hidden_size).to(x_test.device)

            y_pred_test = model(x_test, hidden_state_val, cell_state_val)
            test_loss = criterion(y_pred_test, y_test)
            test_losses.append(test_loss.item())
        model.train()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        prediction = y_pred_test

    prediction = np.vstack(prediction)
    return prediction, losses, test_losses, y_test, price_scaler

prediction, losses, test_losses, y_test, price_scaler = run()

# save data
np_y_pred = prediction.detach().numpy()
df = pd.DataFrame(np_y_pred)
df.to_csv("../model_output/prediction.csv", index=False)

pd.DataFrame(losses).to_csv("../model_output/losses.csv", index=False)
pd.DataFrame(test_losses).to_csv("../model_output/test_losses.csv", index=False)

# plot
v.plot_losses(losses, test_losses)
d_time_series = pd.read_excel('../data_xlsx/d_timeseries.xlsx', index_col=0)
d_time_series['date'] = d_time_series['date'].apply(lambda x: datetime.fromtimestamp(x).date()) # convert from unix timestamp to datetime
date = d_time_series['date']
date = date[:len(prediction)]


#print(prediction.shape)
date = date[-len(y_test):]
v.plot_stocks(date, y_test, prediction, scaler=price_scaler)
