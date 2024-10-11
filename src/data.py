import torch
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def createDataset(data, input_days=30, output_days=7,
                   input_features=['Open', 'High', 'Low', 'Close', 'Volume'],
                   output_feature='Close'):
    """
    Erstellt Eingabe- und Ausgabedatensätze für das neuronale Netzwerk.

    Parameters:
    - data (pd.DataFrame): Historische Börsendaten mit Spalten wie 'Open', 'High', 'Low', 'Close', 'Volume'.
    - input_days (int): Anzahl der Tage, die für die Eingabe verwendet werden.
    - output_days (int): Anzahl der Tage, für die die Vorhersage getroffen werden soll.
    - input_features (list): Liste der Features, die als Eingabe verwendet werden sollen.
    - output_feature (str): Das Feature, das vorhergesagt werden soll.

    Returns:
    - X (np.ndarray): Eingabedaten mit der Form (Anzahl der Samples, input_days * Anzahl der input_features).
    - Y (np.ndarray): Ausgabedaten mit der Form (Anzahl der Samples, output_days).
    """
    X, Y = [], []

    # Gesamte Anzahl der verfügbaren Datenpunkte
    total_days = len(data)

    # Iteriere über die Daten, um Eingabe- und Ausgabesequenzen zu erstellen
    for i in range(total_days - input_days - output_days + 1):
        # Eingabedaten: 30 Tage * 5 Features
        input_data = data[input_features].iloc[i:i + input_days].values
        input_data = input_data.flatten()  # Form: (30 * 5,) = (150,)
        X.append(input_data)

        # Ausgabedaten: 7 Tage * 1 Feature ('Close')
        output_data = data[output_feature].iloc[i + input_days:i + input_days + output_days].values
        Y.append(output_data)

    return np.array(X), np.array(Y)

# Download historical data
data = yf.download("AAPL", period = "10y")

X, Y = createDataset(data)

# Train Test Split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=41)

# reshape data for dimension of model (30 neurons in inputL)
Xtrain = torch.FloatTensor(Xtrain).reshape(-1, 150)
Xtest = torch.FloatTensor(Xtest).reshape(-1, 150)

# convert Y to tensors
Ytrain = torch.FloatTensor(Ytrain)
Ytest = torch.FloatTensor(Ytest)

