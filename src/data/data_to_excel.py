import numpy as np
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
import pandas as pd
from dotenv import load_dotenv
import os

# api key
load_dotenv()
API_KEY = os.getenv("API_KEY")

# directory
directory = '../data_xlsx/'
#directory = '../../data_xlsx/'

def api_raw_data_to_excel():
    # create instances
    api_d_fundamental_data = FundamentalData(key=API_KEY)  # load data once then comment out
    api_d_time_series = TimeSeries(key=API_KEY, output_format='pandas')

    d_time_series, _ = api_d_time_series.get_daily(symbol='AAPL', outputsize='full')
    d_quarterly_income, _ = api_d_fundamental_data.get_income_statement_quarterly(symbol='AAPL')

    d_quarterly_income.to_excel(f'{directory}d_quarterly_income_raw.xlsx', index=True)
    d_time_series.to_excel(f'{directory}d_timeseries_raw.xlsx', index=True)

#api_raw_data_to_excel()

d_quarterly_income = pd.read_excel(f'{directory}d_quarterly_income_raw.xlsx', index_col=0)
d_time_series = pd.read_excel(f'{directory}d_timeseries_raw.xlsx', index_col=0) # 6288 days currently in there
r_time_series = d_time_series
d_time_series = d_time_series.reset_index()

# clean data
d_quarterly_income = d_quarterly_income.drop(columns='depreciation')
d_quarterly_income = d_quarterly_income.drop(columns='reportedCurrency')
d_quarterly_income.replace("None", np.nan)
d_quarterly_income = d_quarterly_income.fillna(0) # replaces ever None with 0
d_quarterly_income['fiscalDateEnding'] = pd.to_datetime(d_quarterly_income['fiscalDateEnding'], errors='coerce')

d_quarterly_income.set_index('fiscalDateEnding', inplace=True)

d_quarterly_income = d_quarterly_income.resample('D').ffill()
d_quarterly_income.reset_index(inplace=True)

d_quarterly_income = d_quarterly_income.iloc[::-1].reset_index(drop=True)

d_quarterly_income.to_excel(f'{directory}d_quarterly_income_TEST.xlsx', index=True)

# set unix time stamps
d_quarterly_income['fiscalDateEnding'] = pd.to_datetime(d_quarterly_income['fiscalDateEnding'])
d_quarterly_income['fiscalDateEnding'] = (d_quarterly_income['fiscalDateEnding'] - pd.Timestamp('1970-01-01')).dt.total_seconds()
d_time_series['date'] = pd.to_datetime(d_time_series['date'])
d_time_series['date'] = (d_time_series['date'] - pd.Timestamp('1970-01-01')).dt.total_seconds()

common_dates = set(d_time_series.index).intersection(set(d_quarterly_income.index))
min_date = max(d_time_series.index.min(), d_quarterly_income.index.min())
max_date = min(d_time_series.index.max(), d_quarterly_income.index.max())

d_quarterly_income.to_excel(f'{directory}d_quarterly_income.xlsx', index=True)
d_time_series.to_excel(f'{directory}d_timeseries.xlsx', index=True)
