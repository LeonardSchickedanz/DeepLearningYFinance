from data import data
import model as model_class
import torch
import visualize as v
import pandas as pd

# training setup
torch.manual_seed(41)
f_input=29
model = model_class.Model(inputL=f_input, hiddenL1=150, hiddenL2=150, hiddenL3=150, outputL=1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# get the data and scaler
X_train, y_train, X_test, y_test, scaler = data.prepare_training_data(data.t_combined)

# print data shapes and sample values
print("Training shapes:", X_train.shape, y_train.shape)
print("Test shapes:", X_test.shape, y_test.shape)
print("\nSample values:")
print("X_train:", X_train[0])
print("y_train:", y_train[0])

# Training loop
model.train()
epochs = 100
losses = []
test_losses = []
prediction = []

for i in range(epochs):
    # training
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss.item())

    # validation
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test)
        test_loss = criterion(y_pred_test, y_test)
        test_losses.append(test_loss.item())
    model.train()

    if i % 100 == 0:
        print(f"Epoch: {i}")
        print(f"Training loss: {loss.item():.4f}")
        print(f"Test loss: {test_loss.item():.4f}")
        print("\n")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# plot
v.plot_losses(losses, test_losses)
print(len(prediction))
d_time_series = pd.read_excel('../data_xlsx/d_timeseries.xlsx', index_col=0)
date = d_time_series['date']
print(len(date))

#v.plot_stocks()
