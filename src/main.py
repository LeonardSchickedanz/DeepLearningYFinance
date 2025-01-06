from datetime import datetime
import numpy as np
from data import data
import model as model_class
import torch
import visualize as v
import pandas as pd

torch.manual_seed(41)
ticker = 'AAPL'
T_COMBINED = data.main(ticker)

model = model_class.LSTMModel(inputL=T_COMBINED.shape[1], hiddenL1=200, hiddenL2=200, hiddenL3=200, outputL=1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def run():
    x_train, x_test, y_train, y_test, main_scaler, price_scaler = data.prepare_training_data(data.T_COMBINED)

    print("x_train:", x_train.shape)
    print("x_test:", x_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    # Geändert: shuffle=False um zeitliche Abhängigkeit zu bewahren
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False)

    epochs = 200
    losses = []
    test_losses = []

    # early stopping parameter
    best_loss = float('inf')
    patience = 10
    no_improve = 0
    best_prediction = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch_x, batch_y in train_loader:
            batch_size = batch_x.size(0)
            hidden_state = torch.zeros(1, batch_size, model.lstm.hidden_size).to(batch_x.device)
            cell_state = torch.zeros(1, batch_size, model.lstm.hidden_size).to(batch_x.device)

            optimizer.zero_grad()

            batch_pred, _ = model(batch_x, hidden_state, cell_state)

            loss = criterion(batch_pred, batch_y)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        model.eval()
        with torch.no_grad():
            batch_size_test = x_test.size(0)
            hidden_state_val = torch.zeros(1, batch_size_test, model.lstm.hidden_size).to(x_test.device)
            cell_state_val = torch.zeros(1, batch_size_test, model.lstm.hidden_size).to(x_test.device)

            y_pred_test, _ = model.forward(x_test, hidden_state_val, cell_state_val)
            test_loss = criterion(y_pred_test, y_test)
            test_losses.append(test_loss.item())

            # early stopping logic
            if test_loss < best_loss:
                best_loss = test_loss
                best_prediction = y_pred_test.detach().numpy().flatten()
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"Epoch {epoch}: Train Loss = {avg_loss}, Test Loss = {test_loss.item()}")

    # descaling
    final_prediction = price_scaler.inverse_transform(best_prediction.reshape(-1, 1)).flatten()
    y_test_descaled = price_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    return final_prediction, losses, test_losses, y_test_descaled, main_scaler, price_scaler


prediction, losses, test_losses, y_test, main_scaler, price_scaler = run()

# saving model data
torch.save(model.state_dict(), 'best_model.pth')

#print(type(prediction))
#print("prediction[0]", prediction[0])

# save data
np_y_pred = np.array(prediction)
df = pd.DataFrame(np_y_pred)
df.to_csv("../model_output/prediction.csv", index=False)

pd.DataFrame(losses).to_csv("../model_output/losses.csv", index=False)
pd.DataFrame(test_losses).to_csv("../model_output/test_losses.csv", index=False)

# plot
v.plot_losses(losses, test_losses)
d_time_series = pd.read_excel(r'C:\Users\LeonardSchickedanz\PycharmProjects\PredictStockPrice\data\processed\d_timeseries.xlsx', index_col=0)
d_time_series['date'] = d_time_series['date'].apply(lambda x: datetime.fromtimestamp(x).date()) # convert from unix timestamp to datetime
date = d_time_series['date']
date = date[:len(prediction)]

date = date[-len(y_test):]
v.plot_stocks(date, y_test, prediction, scaler=price_scaler)

def evaluate_prediction(actual, forecast):
    diff = np.array(actual) - np.array(forecast)
    mae = np.mean(np.abs(diff))
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100

    r_squared = 1 - (np.sum((actual - forecast) ** 2) / np.sum((actual - np.mean(actual)) ** 2))
    print(f'mean absolut error: {mae}')
    print(f'mean squared error: {mse}')
    print(f'root mean squared error: {rmse}')
    print(f'R-Squared: {r_squared}')
    print(f'mean absolute percentage error: {mape}')

evaluate_prediction(y_test, prediction)
