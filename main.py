import yfinance as yf
import pandas as pd
import torch as torch

ticker_symbol = "AAPL"
ticker = yf.Ticker(ticker_symbol)
df = ticker.history(period="max")
print(df.tail())