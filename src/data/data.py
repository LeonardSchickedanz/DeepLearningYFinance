import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from src.data import data_to_excel

# t_ = tensor
# d_ = raw data
# f_ = features

top_100_sp500 = (
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
    )

# when running from main.py
directory = '../data_xlsx/'
directory_processed = '../data/processed/'

# when running from data.py
#directory = '../../data_xlsx/'
#directory_processed = '../../data/processed/'


DATA_LIST = (
    pd.read_excel(f'{directory_processed}d_real_gdp.xlsx', index_col=0),
    pd.read_excel(f'{directory_processed}d_real_gdp_per_capita.xlsx', index_col=0),
    pd.read_excel(f'{directory_processed}d_federal_funds_rate.xlsx', index_col=0),
    pd.read_excel(f'{directory_processed}d_cpi.xlsx', index_col=0),
    pd.read_excel(f'{directory_processed}d_inflation.xlsx', index_col=0),
    pd.read_excel(f'{directory_processed}d_retail_sales.xlsx', index_col=0),
    pd.read_excel(f'{directory_processed}d_durables.xlsx', index_col=0),
    pd.read_excel(f'{directory_processed}d_unemployment.xlsx', index_col=0),
    pd.read_excel(f'{directory_processed}d_nonfarm_payroll.xlsx', index_col=0),
    pd.read_excel(f'{directory_processed}d_quarterly_income.xlsx', index_col=0),
    pd.read_excel(f'{directory_processed}d_timeseries.xlsx', index_col=0),
)

def read_and_merge_dataframes(dataframes):
    d_merged = pd.concat(dataframes, axis=1)
    d_merged = d_merged.loc[:, ~d_merged.columns.duplicated()]
    d_merged = d_merged.drop(columns='value')
    return d_merged

def prepare_training_data(tensor, look_back_days=365, forecast_horizon=30, closed_price_column=36):
    tensor = torch.flip(tensor, [0])
    size_rows = tensor.size(0)
    x = []
    y = []

    price_scaler = MinMaxScaler(feature_range=(0, 1))
    rest_scaler = MinMaxScaler(feature_range=(0, 1))

    column_to_plot = tensor[:, closed_price_column]
    # scale the priced column separately
    priced_column_data = tensor[:, closed_price_column].unsqueeze(1)
    scaled_priced_column = torch.tensor(price_scaler.fit_transform(priced_column_data.numpy()), dtype=tensor.dtype)

    # scale the rest of the columns
    rest_columns = torch.cat([tensor[:, :closed_price_column], tensor[:, closed_price_column + 1:]], dim=1)
    scaled_rest_columns = torch.tensor(rest_scaler.fit_transform(rest_columns.numpy()), dtype=tensor.dtype)

    # reconstruct the scaled tensor
    scaled_tensor = torch.cat([scaled_rest_columns[:, :closed_price_column],
                               scaled_priced_column,
                               scaled_rest_columns[:, closed_price_column:]], dim=1)

    for idx in range(size_rows - look_back_days - forecast_horizon):
        # X-Block: look_back_days Reihen
        x_block = scaled_tensor[idx:idx + look_back_days, :]

        # Y-Wert: Vorhersagewert nach look_back_days + forecast_horizon
        y_value = scaled_tensor[idx + look_back_days + forecast_horizon, closed_price_column]
        x.append(x_block)
        y.append(y_value)

    x = torch.stack(x)
    y = torch.stack(y)

    split_ratio = 0.8
    split_index = int(len(x) * split_ratio)
    x_train = x[:split_index]
    y_train = y[:split_index]
    x_test = x[split_index:]
    y_test = y[split_index:]
    y_train = y_train.view(-1, 1)
    y_test = y_test.view(-1, 1)

    return x_train, x_test, y_train, y_test, rest_scaler, price_scaler

def main(ticker, call_data_to_excel_main = False):

    if call_data_to_excel_main is True:
        data_to_excel.main(ticker)

    d_combined = read_and_merge_dataframes(DATA_LIST)
    d_combined.to_excel(f'{directory_processed}d_combined.xlsx') # for debugging

    t_combined = torch.tensor(d_combined.values).float()
    return t_combined
