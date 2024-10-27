import data
import model as model_class
import torch
import matplotlib.pyplot as plt


def plot_losses(losses, test_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_predictions(y_true, y_pred, scaler, title="Training" or "Test"):
    """
    Plot actual vs predicted stock prices with inverse transform
    """
    # Reshape the data for inverse transform
    y_true_reshaped = y_true.numpy().reshape(-1, 1)
    y_pred_reshaped = y_pred.numpy().reshape(-1, 1)

    # Inverse transform the data
    y_true_orig = scaler.inverse_transform(y_true_reshaped)
    y_pred_orig = scaler.inverse_transform(y_pred_reshaped)

    plt.figure(figsize=(15, 6))
    plt.plot(y_true_orig, label='Actual', color='blue')
    plt.plot(y_pred_orig, label='Predicted', color='red', alpha=0.7)
    plt.title(f'{title} Set: Actual vs Predicted Close Prices')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)
    plt.show()


def calculate_metrics(y_true, y_pred, scaler):
    """
    Calculate metrics using inverse transformed data
    """
    # Inverse transform the data
    y_true_orig = scaler.inverse_transform(y_true.numpy().reshape(-1, 1))
    y_pred_orig = scaler.inverse_transform(y_pred.numpy().reshape(-1, 1))

    # Convert back to tensors
    y_true_orig = torch.FloatTensor(y_true_orig)
    y_pred_orig = torch.FloatTensor(y_pred_orig)

    mse = torch.nn.MSELoss()(y_true_orig, y_pred_orig)
    mae = torch.nn.L1Loss()(y_true_orig, y_pred_orig)

    # Calculate R²
    y_mean = torch.mean(y_true_orig)
    ss_tot = torch.sum((y_true_orig - y_mean) ** 2)
    ss_res = torch.sum((y_true_orig - y_pred_orig) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return {
        'MSE': mse.item(),
        'MAE': mae.item(),
        'R²': r2.item()
    }


# Training setup
torch.manual_seed(41)
model = model_class.Model(inputL=28, hiddenL1=150, hiddenL2=150, hiddenL3=150, outputL=1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Get the data and scaler
X_train, y_train, X_test, y_test, scaler = data.prepare_training_data(data.t_combined)

# Print data shapes and sample values
print("Training shapes:", X_train.shape, y_train.shape)
print("Test shapes:", X_test.shape, y_test.shape)
print("\nSample values:")
print("X_train:", X_train[0])
print("y_train:", y_train[0])

# Training loop
model.train()
epochs = 1000
losses = []
test_losses = []

for i in range(epochs):
    # Training
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss.item())

    # Validation
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

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Plot losses
plot_losses(losses, test_losses)

# Final evaluation
model.eval()
with torch.no_grad():
    train_predictions = model(X_train)
    test_predictions = model(X_test)

    # Plot predictions with inverse transformed data
    plot_predictions(y_train, train_predictions, scaler, "Training")
    plot_predictions(y_test, test_predictions, scaler, "Test")

    # Calculate and print metrics
    print("\nTraining Set Metrics:")
    train_metrics = calculate_metrics(y_train, train_predictions, scaler)
    for metric, value in train_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nTest Set Metrics:")
    test_metrics = calculate_metrics(y_test, test_predictions, scaler)
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")