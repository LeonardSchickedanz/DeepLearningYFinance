import torch
import yfinance as yf
import numpy as np
from sklearn.model_selection import train_test_split

def createDataset(data, input_days=30, output_days=7, input_features=['Open', 'High', 'Low', 'Close', 'Volume'], output_feature='Close'):

    x,y = [], []

    input_data = data[input_features].values
    output_data = data[output_feature].values
    print(f"output_data: {output_data}, shape: {output_data.shape}")

    total_days = len(data)

    # Iteriere über die Daten, um Eingabe- und Ausgabesequenzen zu erstellen
    for i in range(total_days - input_days - output_days + 1):

        x.append(input_data[i:i + input_days].flatten())  # select windows of 30 days (from i to i + 30) with 5 features

        y.append(output_data[i + input_days:i + input_days + output_days]) # select window of 7 days, directly behind the 30 day window of x (for 1 output feature)

    return np.array(x), np.array(y)

# Download historical data
data = yf.download("AAPL", period = "10y")

x, y = createDataset(data)

# train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=41)

# reshape x for dimension of model (30 days * 5 features)
x_train = torch.FloatTensor(x_train).reshape(-1, 150)
x_test = torch.FloatTensor(x_test).reshape(-1, 150)

# convert y to tensors
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

