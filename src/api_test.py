import os

from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.timeseries import TimeSeries
from dotenv import load_dotenv
import yfinance as yf

# API KEY
load_dotenv()
API_KEY = os.getenv("API_KEY")

directory_raw = r'C:\\Users\\LeonardSchickedanz\\PycharmProjects\\PredictStockPrice\data\raw'
directory_processed = r'C:\\Users\\LeonardSchickedanz\\PycharmProjects\\PredictStockPrice\\data\\processed'

def test_time_series(ticker):

    d_time_series = yf.download(ticker,
                                start="2000-01-01",
                                end="2024-12-17",
                                interval="1d",
                                auto_adjust=True)

    d_time_series.index = d_time_series.index.tz_localize(None)
    d_time_series.to_excel(fr'{directory_raw}\\d_timeseries_raw.xlsx', index=True)

#test_time_series('NFLX')

def api_raw_data_to_excel(ticker):
    api_d_fundamental_data = FundamentalData(key=API_KEY, output_format='pandas')
    api_d_time_series = TimeSeries(key=API_KEY, output_format='pandas')
    # d_time_series, _ = api_d_time_series.get_daily_adjusted(symbol='ticker', outputsize='full')
    d_quarterly_income, _ = api_d_fundamental_data.get_income_statement_quarterly(symbol=ticker)

    d_time_series = yf.download(ticker,
                                start="2000-01-01",
                                end="2024-12-17",
                                interval="1d",
                                auto_adjust=True)

    d_time_series.index = d_time_series.index.tz_localize(None)

    d_quarterly_income.to_excel(fr'{directory_raw}\\d_quarterly_income_raw.xlsx', index=True)
    d_time_series.to_excel(fr'{directory_raw}\\d_timeseries_raw.xlsx', index=True)

api_raw_data_to_excel('NFLX')