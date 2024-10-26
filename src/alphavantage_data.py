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

pd.set_option('display.max_columns', None)  # Zeigt alle Spalten an
pd.set_option('display.max_rows', None)     # Zeigt alle Zeilen an
pd.set_option('display.width', None)        # Passt die Breite an das Terminal an

# create instances
api_fundamental_data = FundamentalData(key=API_KEY) #load data once then comment out
api_time_series_data = TimeSeries(key=API_KEY, output_format='pandas')

api_fundamental_data.to_excel('../data/fundamental_data.xlsx', index=False)
api_time_series_data.to_excel('../data/raw_data.xlsx', index=False)

data_quarterly_income = pd.read_excel('../data/fundamental_data.xlsx')
time_series_data = pd.read_excel('../data/raw_data.xlsx')

time_series_data, _ = time_series_data.get_daily(symbol='AAPL', outputsize='compact') # 100 days

data_quarterly_income, _ = data_quarterly_income.get_income_statement_quarterly(symbol='AAPL')
latest_quarters = data_quarterly_income.head(8) # last 8 quarterly earnings

# create tensors
