import model
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import torch
import data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

print("test")
# Pick manual seed
torch.manual_seed(41)

# Create model
modelInstance = model.Model()

# set criterion
criterion = torch.nn.MSELoss()

# choose adam optimizer, lr = learning rate (if error doesn't go down after a bunch of iterations lower lr)
optimizer = torch.optim.Adam(modelInstance.parameters(), lr=0.001)  # parameters are layers

# Train model
# Epochs (one run through all the training data)

Xtrain = data.Xtrain
Ytrain = data.Ytrain

modelInstance.train() # set model to trainingsmode

epochs = 100
losses = []
for i in range(epochs):
    # go forward and get a prediction
    Ypred = modelInstance(Xtrain)  # get predicted results

    # measure loss/error
    loss = criterion(Ypred, Ytrain)  # predicted values vs the Ytrain

    # keep track of losses
    losses.append(loss.item())

    # print every 10 epoch
    if i % 10 == 0:
        print(f"Epoch: {i} loss: {loss.item():.4f}")

    # back propagation
    optimizer.zero_grad()  # reset gradients
    loss.backward()  # calculate gradients
    optimizer.step()  # update weights

# 7. Plot der Trainingsverluste
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss über Epochen')
plt.legend()
plt.grid(True)
plt.show()

# 8. Evaluation auf Trainingsdaten und Visualisierung der Vorhersagen
modelInstance.eval()  # Setze Modell in Evaluationsmodus
with torch.no_grad():
    YtrainPred = modelInstance(Xtrain)

    # Konvertiere Tensoren zu NumPy-Arrays
    YtrainPred = YtrainPred.numpy()
    YtrainTrue = Ytrain.numpy()

    # Invertiere die Normalisierung für 'Close'-Preise (falls normalisiert)
    # Annahme: 'Close' wurde mit MinMaxScaler normalisiert. Falls nicht, überspringe diesen Schritt.
    # Falls du separate Scaler für Eingaben und Ausgaben verwendest, passe dies entsprechend an.
    close_scaler = MinMaxScaler(feature_range=(0, 1))
    # Fit den Scaler auf die 'Close'-Preise aus den ursprünglichen Daten
    original_data = yf.download("AAPL", period="10y")
    close_scaler.fit(original_data[['Close']])

    # Invertiere die Normalisierung für die Vorhersagen und die echten Werte
    Y_train_pred_inverse = close_scaler.inverse_transform(YtrainPred)
    Y_train_true_inverse = close_scaler.inverse_transform(YtrainTrue)

    # Wähle eine Anzahl von Beispielen zum Plotten
    num_examples = 5  # Anzahl der zu plottenden Beispiele

    for i in range(num_examples):
        plt.figure(figsize=(10, 4))
        plt.plot(range(1, 8), Y_train_true_inverse[i], marker='o', label='Tatsächlich')
        plt.plot(range(1, 8), Y_train_pred_inverse[i], marker='x', label='Vorhergesagt')
        plt.title(f'Sample {i + 1}: Tatsächlich vs Vorhergesagt')
        plt.xlabel('Tag')
        plt.ylabel('Close Preis')
        plt.legend()
        plt.grid(True)
        plt.show()