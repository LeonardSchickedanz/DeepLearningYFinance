import torch
import yfinance as yf
import numpy as np
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import os
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
from pprint import pprint
import matplotlib.pyplot as plt
import pandas as pd

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

# create instances
api_d_fundamental_data = FundamentalData(key=API_KEY) #load data once then comment out
api_d_time_series = TimeSeries(key=API_KEY, output_format='pandas')

d_quarterly_income, _ = api_d_fundamental_data.get_income_statement_quarterly(symbol='AAPL')
d_time_series, _ = api_d_time_series.get_daily(symbol='AAPL', outputsize='compact')

d_quarterly_income.to_excel('../data/d_fundamental.xlsx', index=False)
d_time_series.to_excel('../data/data_timeseries.xlsx', index=False)

d_quarterly_income = pd.read_excel('../data/d_fundamental.xlsx')
d_time_series = pd.read_excel('../data/data_timeseries.xlsx')

d_time_series, _ = d_time_series.get_daily(symbol='AAPL', outputsize='compact') # 100 days

d_quarterly_income, _ = d_quarterly_income.get_income_statement_quarterly(symbol='AAPL')
d_latest_quarters = d_quarterly_income.head(8) # last 8 quarterly earnings

# create tensors
f_time_series = 6 # open, high, low, close, adj close, volume
f_quarterly_income = 20

# remove date of data
d_time_series = d_time_series.drop(columns=['date'])
d_quarterly_income = d_quarterly_income.drop(columns='Fiscal Date Ending')

# create tensors
t_time_series = torch.tensor(d_time_series.values).float()
t_quarterly_income = torch.tensor(d_quarterly_income.values).float()

t_quarterly_income = t_quarterly_income.unsqueeze(0).expand(100, -1)  # transform to (100, 20) for cat
t_combined = torch.cat((t_time_series, t_quarterly_income), dim=1) #(100, 26)

# now call model with 260 input neurons
