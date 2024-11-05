import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# t_ = tensor
# d_ = raw data_xlsx
# f_ = features
# bs_ = batch size

#get_income_statement, get_balance_sheet, get_cash_flow

top_100_sp500 = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "BRK.B", "META", "TSLA", "V", "JNJ",
    "WMT", "PG", "XOM", "MA", "LLY", "UNH", "HD", "AVGO", "JPM", "CVX",
    "MRK", "ABBV", "PEP", "KO", "COST", "ORCL", "MS", "MCD", "TMO", "CSCO",
    "ADBE", "NFLX", "NKE", "AMD", "CRM", "LIN", "VZ", "WFC", "RTX", "INTC",
    "CMCSA", "SCHW", "NEE", "TXN", "PM", "HON", "UNP", "IBM", "UPS", "QCOM",
    "ELV", "AMGN", "PLD", "LOW", "MDT", "SPGI", "INTU", "AMT", "DHR", "BMY",
    "CAT", "GE", "GS", "NOW", "MO", "T", "BKNG", "SYK", "BLK", "AXP",
    "GILD", "ISRG", "ZTS", "ADP", "MDLZ", "DE", "CB", "PGR", "CCI", "CI",
    "EQIX", "DUK", "SO", "MMC", "USB", "C", "REGN", "TGT", "BDX", "APD",
    "ITW", "ADI", "HUM", "ETN", "NSC", "PNC", "WM", "AON", "ADSK", "MU",
    "MET", "VRTX", "EL", "AEP", "SLB", "TRV", "EOG", "SHW"
]

# settings for seeing all data_xlsx in the terminal
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

# read data from excel
d_time_series = pd.read_excel(r'C:\Users\LeonardSchickedanz\PycharmProjects\PredictStockPrice\data_xlsx\d_timeseries.xlsx', index_col=0)
d_quarterly_income = pd.read_excel(r'C:\Users\LeonardSchickedanz\PycharmProjects\PredictStockPrice\data_xlsx\d_quarterly_income.xlsx', index_col=0)

print(d_time_series.head())
print(d_quarterly_income.head()) # Warum ist das anders?

f_time_series = d_time_series.shape[1] # date, open, high, low, close, volume
bs_time_series = d_time_series.shape[0] # 6
f_quarterly_income = d_quarterly_income.shape[1]
bs_quarterly_income = d_quarterly_income.shape[0] # 24

f_input = f_time_series + f_quarterly_income # 30

# tensors
t_time_series = torch.tensor(d_time_series.values).float()
t_time_series = t_time_series[-5937:]  # last 5850 days, Shape: (5937, 6)

t_quarterly_income = torch.tensor(d_quarterly_income.values).float()
t_quarterly_income = t_quarterly_income[-5937:] # Shape: (5937, 24)

t_combined = torch.cat((t_time_series, t_quarterly_income), dim=1)  # Shape: (5937, 30)

def prepare_training_data(tensor, forecast_horizon=30):

    #  scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_np = tensor.numpy()
    scaled_data = scaler.fit_transform(data_np)
    scaled_tensor = torch.FloatTensor(scaled_data)

    # total number of samples we can create
    n_samples = len(scaled_tensor) - forecast_horizon

    # Prepare X (features) and y (targets)
    X = scaled_tensor[:n_samples]  # Input features for each day
    y = scaled_tensor[forecast_horizon:, 3:4]  # select close price (4th column)

    # Calculate split point (80/20)
    split_idx = int(n_samples * 0.8)

    # Split into train and test
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]

    # Print shapes and sample values for verification
    print(f"Training samples: {X_train.shape}")
    print(f"Test samples: {X_test.shape}")
    print(f"\nSample scaled values:")
    print(f"X_train first row: {X_train[0]}")
    print(f"y_train first value: {y_train[0]}")

    return X_train, y_train, X_test, y_test, scaler

# Plot erstellen
#plt.figure(figsize=(10, 5))
#plt.plot(d_time_series.index, d_time_series['4. close'], label="Schlusskurs")
#print(t_combined[0])
#plt.show()


