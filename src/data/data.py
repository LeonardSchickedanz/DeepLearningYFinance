import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# t_ = tensor
# d_ = raw data
# f_ = features
# bs_ = batch size

#get_income_statement, get_balance_sheet, get_cash_flow

FORECASTHORIZON = 30

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
#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
#pd.set_option('display.width', None)

# read data from excel
d_time_series = pd.read_excel('../data_xlsx/d_timeseries.xlsx', index_col=0)
d_quarterly_income = pd.read_excel('../data_xlsx/d_quarterly_income.xlsx', index_col=0)

f_time_series = d_time_series.shape[1] # date, open, high, low, close, volume
bs_time_series = d_time_series.shape[0] # 6
f_quarterly_income = d_quarterly_income.shape[1]
bs_quarterly_income = d_quarterly_income.shape[0] # 24

f_input = f_time_series + f_quarterly_income # 29

# Dann die Tensoren erstellen
min_date = max(d_time_series.index.min(), d_quarterly_income.index.min())
max_date = min(d_time_series.index.max(), d_quarterly_income.index.max())

d_time_series_aligned = d_time_series[min_date:max_date]
d_quarterly_income_aligned = d_quarterly_income[min_date:max_date]

t_time_series = torch.tensor(d_time_series_aligned.values).float()
t_quarterly_income = torch.tensor(d_quarterly_income_aligned.values).float()

# remove date from tensor
t_quarterly_income_without_date = t_quarterly_income[:, 1:]  # Alle Spalten außer der ersten (Datum)

# combine tensors
t_combined = torch.cat((t_time_series, t_quarterly_income_without_date), dim=1)

# Überprüfen der Shape
print("t_combined:")
print(t_combined.shape)


def prepare_training_data(tensor, look_back_days=365, forecast_horizon=30, closed_price_index=4):
    # Scaler für verschiedene Spalten
    full_scaler = MinMaxScaler(feature_range=(0, 1))
    price_scaler = MinMaxScaler(feature_range=(0, 1))

    # Skalieren der Spalten separat
    scaled_tensor = tensor.clone().float()  # Konvertiere zu Float

    # Skalierung mit übergebener Closed Price Index
    scaled_tensor[:, 1:closed_price_index + 1] = torch.tensor(full_scaler.fit_transform(tensor[:, 1:closed_price_index + 1]))
    scaled_tensor[:, -1] = torch.tensor(price_scaler.fit_transform(tensor[:, -1:].reshape(-1, 1))).flatten()

    size_rows = scaled_tensor.size(0)
    x = []
    y = []

    for idx in range(size_rows - look_back_days - forecast_horizon):
        x_block = scaled_tensor[idx:idx + look_back_days, :]
        y_row = scaled_tensor[idx + look_back_days + forecast_horizon, closed_price_index]

        x.append(x_block)
        y.append(y_row)

    # Konvertieren zu Tensoren
    x = torch.stack(x)
    y = torch.stack(y)

    # Train-Test-Split (80:20)
    train_size = int(0.8 * len(x))
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return x_train, x_test, y_train, y_test, full_scaler, price_scaler




