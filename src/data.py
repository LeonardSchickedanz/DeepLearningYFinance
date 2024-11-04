import torch
import yfinance as yf
import numpy as np
from dotenv import load_dotenv
import os
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import visualize as v
import matplotlib.pyplot as plt

# t_ = tensor
# d_ = raw data
# f_ = features

load_dotenv()
API_KEY = os.getenv("API_KEY")

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

# settings for seeing all data in the terminal
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

def api_raw_data_to_excel():
    # create instances
    api_d_fundamental_data = FundamentalData(key=API_KEY)  # load data once then comment out
    api_d_time_series = TimeSeries(key=API_KEY, output_format='pandas')

    d_time_series, _ = api_d_time_series.get_daily(symbol='AAPL', outputsize='full')
    d_quarterly_income, _ = api_d_fundamental_data.get_income_statement_quarterly(symbol='AAPL')

    d_quarterly_income.to_excel('../data/d_quarterly_income_raw.xlsx', index=True)
    d_time_series.to_excel('../data/d_timeseries_raw.xlsx', index=True)

#api_raw_data_to_excel()

d_quarterly_income = pd.read_excel('../data/d_quarterly_income_raw.xlsx', index_col=0)
d_time_series = pd.read_excel('../data/d_timeseries_raw.xlsx', index_col=0) # 6288 days currently in there
r_time_series = d_time_series

# clean data
d_quarterly_income = d_quarterly_income.drop(columns='depreciation')
d_quarterly_income = d_quarterly_income.drop(columns='reportedCurrency')
d_quarterly_income.replace("None", np.nan)
d_quarterly_income = d_quarterly_income.fillna(0) # replaces ever None with 0

d_quarterly_income.to_excel('../data/d_quarterly_income.xlsx', index=True)
d_time_series.to_excel('../data/d_timeseries.xlsx', index=True)

d_time_series = pd.read_excel('../data/d_timeseries.xlsx', index_col=0)
d_quarterly_income = pd.read_excel('../data/d_quarterly_income.xlsx', index_col=0)

def stretch_quarterly_income(data):
    # Erstelle einen DataFrame aus den Daten
    df = pd.DataFrame(data)

    # Konvertiere das fiscalDateEnding-Datum in ein datetime-Objekt
    df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])

    # Erstelle eine Liste mit allen Monaten, die abgedeckt werden
    all_months = pd.date_range(start=df['fiscalDateEnding'].min(), end=df['fiscalDateEnding'].max(), freq='ME')

    # Initialisiere einen leeren DataFrame, um die Daten zu speichern
    stretched_data = pd.DataFrame()

    # Iteriere über die Quartale und erstelle die monatlichen Datensätze
    for _, row in df.iterrows():
        fiscal_date = row['fiscalDateEnding']
        monthly_data = pd.DataFrame({col: [row[col]] for col in df.columns},
                                    index=[fiscal_date + pd.Timedelta(days=i) for i in range(3)])
        stretched_data = pd.concat([stretched_data, monthly_data], ignore_index=True)

    print(stretched_data)

stretch_quarterly_income(d_quarterly_income)

# create time series tensor
f_time_series = d_time_series.shape[1] # open, high, low, close, volume
bs_time_series = d_time_series.shape[0]
f_quarterly_income = d_quarterly_income.shape[1]
bs_quarterly_income = d_quarterly_income.shape[0]

f_input = f_time_series + f_quarterly_income # 28

t_time_series = torch.tensor(d_time_series.values).float()
t_time_series = t_time_series[-5850:]  # last 5850 days, Shape: (5850, 5)

# create quarterly income tensor
# transform date to right format
#d_quarterly_income['fiscalDateEnding'] = pd.to_datetime(d_quarterly_income['fiscalDateEnding'])
#d_quarterly_income['fiscalDateEnding'] = d_quarterly_income['fiscalDateEnding'].astype(int) // 10**9  # Unix timestamp

#print(d_time_series)
#print(d_quarterly_income)

t_quarterly_income = torch.tensor(d_quarterly_income.values).float() # Shape: (65, 23)
t_quarterly_income = t_quarterly_income[-65:]

# First expand t_quarterly_income to match batch size of t_time_series
repeats = 5850 // 65

# Expand quarterly income data
t_quarterly_income_expanded = t_quarterly_income.repeat((repeats, 1))

# Now combine both tensors along feature dimension (dim=1)
t_combined = torch.cat([t_time_series, t_quarterly_income_expanded], dim=1)

# Verify shapes
#print("\nTime series shape:", t_time_series.size())  # Should be (5850, 5)
#print("Expanded quarterly shape:", t_quarterly_income_expanded.size())  # Should be (5850, 23)
#print("Combined tensor shape:", t_combined.size())  # Should be (5850, 28)

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
#    print(f"Training samples: {X_train.shape}")
#    print(f"Test samples: {X_test.shape}")
#    print(f"\nSample scaled values:")
#    print(f"X_train first row: {X_train[0]}")
#    print(f"y_train first value: {y_train[0]}")

    return X_train, y_train, X_test, y_test, scaler

#print(t_time_series)
d_time_series.index = d_time_series.index /365

# Plot erstellen
plt.figure(figsize=(10, 5))
plt.plot(d_time_series.index, d_time_series['4. close'], label="Schlusskurs")

plt.show()


