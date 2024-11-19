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


def prepare_training_data(tensor, look_back_days=365, forecast_horizon=30, price_column=4):
    size_rows = tensor.size(0)
    x = []
    y = []

    price_scaler = MinMaxScaler(feature_range=(0, 1))
    rest_scaler = MinMaxScaler(feature_range=(0, 1))

    # Scale the priced column separately
    priced_column_data = tensor[:, price_column].unsqueeze(1)
    scaled_priced_column = torch.tensor(price_scaler.fit_transform(priced_column_data.numpy()), dtype=tensor.dtype)

    # Scale the rest of the columns
    rest_columns = torch.cat([tensor[:, :price_column], tensor[:, price_column+1:]], dim=1)
    scaled_rest_columns = torch.tensor(rest_scaler.fit_transform(rest_columns.numpy()), dtype=tensor.dtype)

    # Reconstruct the scaled tensor
    scaled_tensor = torch.cat([scaled_rest_columns[:, :price_column],
                                scaled_priced_column,
                                scaled_rest_columns[:, price_column:]], dim=1)

    for idx in range(size_rows - look_back_days - forecast_horizon):
        # X-Block: look_back_days Reihen
        x_block = scaled_tensor[idx:idx + look_back_days, :]

        # Y-Wert: Vorhersagewert nach look_back_days + forecast_horizon
        y_value = scaled_tensor[idx + look_back_days + forecast_horizon, price_column]
        x.append(x_block)
        y.append(y_value)

    # Konvertieren zu Tensoren, falls noch nicht geschehen

    x = torch.stack(x)
    y = torch.stack(y)

    split_ratio = 0.8
    split_index = int(len(x) * split_ratio)
    x_train = x[:split_index]
    y_train = y[:split_index]
    x_test = x[split_index:]
    y_test = y[split_index:]

    return x_train, x_test, y_train, y_test, rest_scaler, price_scaler




