import json
from token import EQUAL

import numpy as np
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
import pandas as pd
from dotenv import load_dotenv
import os
import requests

# api key
load_dotenv()
API_KEY = os.getenv("API_KEY")

# directory
#directory = '../data_xlsx/'
directory_raw = '../../data/raw/'
directory_processed = '../../data/processed/'

# RECEIVE DATA

def api_raw_data_to_excel():

    api_d_fundamental_data = FundamentalData(key=API_KEY, output_format='pandas')
    api_d_time_series = TimeSeries(key=API_KEY, output_format='pandas')

    d_time_series, _ = api_d_time_series.get_daily(symbol='AAPL', outputsize='full')
    d_quarterly_income, _ = api_d_fundamental_data.get_income_statement_quarterly(symbol='AAPL')

    d_quarterly_income.to_excel(f'{directory_raw}d_quarterly_income_raw.xlsx', index=True)
    d_time_series.to_excel(f'{directory_raw}d_timeseries_raw.xlsx', index=True)

def economic_indicators_to_excel():
    main_url = 'https://www.alphavantage.co/query?function='
    api_key_url = f'&apikey={API_KEY}'

    url_list = [
        'REAL_GDP&interval=annual',
        'REAL_GDP_PER_CAPITA',
        'TREASURY_YIELD&interval=daily&maturity=10year',
        'FEDERAL_FUNDS_RATE&interval=monthly',
        'CPI&interval=monthly',
        'INFLATION',
        'RETAIL_SALES',
        'DURABLES',
        'UNEMPLOYMENT',
        'NONFARM_PAYROLL'
    ]

    economic_indicators = [
        'd_real_gdp',
        'd_real_gdp_per_capita',
        'd_treasury_yield',
        'd_federal_funds_rate',
        'd_cpi',
        'd_inflation',
        'd_retail_sales',
        'd_durables',
        'd_unemployment',
        'd_nonfarm_payroll'
    ]

    dataframes = []

    for idx1, u in enumerate(url_list):
        url =  f'{main_url}{u}{api_key_url}'
        r = requests.get(url)
        data = r.json()

        dates = []
        values = []

        for idx2 in range(0, len(data['data'])):
            dates.append(data['data'][idx2]['date'])
            values.append(data['data'][idx2]['value'])

        assert len(dates) == len(values)

        df = pd.DataFrame({
            "date": dates,
            "value": values
        })
        dataframes.append(df)

        df.to_excel(f'{directory_raw}{economic_indicators[idx1]}_raw.xlsx', index=True)

    return dataframes

#economic_indicators_to_excel()
#api_raw_data_to_excel()

d_cpi = pd.read_excel(f'{directory_raw}d_cpi_raw.xlsx', index_col=0)
d_durables = pd.read_excel(f'{directory_raw}d_durables_raw.xlsx', index_col=0)
d_federal_funds_rate = pd.read_excel(f'{directory_raw}d_federal_funds_rate_raw.xlsx', index_col=0)
d_inflation = pd.read_excel(f'{directory_raw}d_inflation_raw.xlsx', index_col=0)
d_nonfarm_payroll = pd.read_excel(f'{directory_raw}d_nonfarm_payroll_raw.xlsx', index_col=0)
d_quarterly_income = pd.read_excel(f'{directory_raw}d_quarterly_income_raw.xlsx', index_col=0)
d_real_gdp_per_capita = pd.read_excel(f'{directory_raw}d_real_gdp_per_capita_raw.xlsx', index_col=0)
d_real_gdp = pd.read_excel(f'{directory_raw}d_real_gdp_raw.xlsx', index_col=0)
d_retail_sales = pd.read_excel(f'{directory_raw}d_retail_sales_raw.xlsx', index_col=0)
d_time_series = pd.read_excel(f'{directory_raw}d_timeseries_raw.xlsx', index_col=0)
d_treasury_yield = pd.read_excel(f'{directory_raw}d_treasury_yield_raw.xlsx', index_col=0)
d_unemployment = pd.read_excel(f'{directory_raw}d_unemployment_raw.xlsx', index_col=0)

# CLEAN DATA
d_time_series = d_time_series.reset_index()
d_quarterly_income = d_quarterly_income.drop(columns='depreciation')
d_quarterly_income = d_quarterly_income.drop(columns='reportedCurrency')
d_quarterly_income.replace("None", np.nan)
d_quarterly_income = d_quarterly_income.fillna(0) # replaces every None with 0
d_quarterly_income.rename(columns={'fiscalDateEnding': 'date'}, inplace=True)
d_quarterly_income['date'] = pd.to_datetime(d_quarterly_income['fiscalDateEnding'], errors='coerce')

#-----------------------------------------------------------------------------------------------------------------------
# stretch quarterly income
d_quarterly_income.set_index('date', inplace=True)
d_quarterly_income = d_quarterly_income.resample('D').ffill()
d_quarterly_income.reset_index(inplace=True)

d_quarterly_income = d_quarterly_income.iloc[::-1].reset_index(drop=True)

# timeseries
max_date_time_series = d_time_series['date'].max()
d_time_series.set_index('date', inplace=True)
d_time_series.sort_index(inplace=True)

date_range = pd.date_range(start=d_time_series.index.min(), end=d_time_series.index.max(), freq='D')
d_time_series = d_time_series.reindex(date_range)

# forward fill
d_time_series = d_time_series.ffill()
current_date = d_quarterly_income['date'].max()

# general
#MAX_DATE = max()

first_row = d_quarterly_income.iloc[0].copy()

while current_date < max_date_time_series:
    current_date += pd.Timedelta(days=1)
    new_row = first_row.copy()
    new_row['date'] = current_date
    # Füge die neue Zeile oben an den DataFrame an
    d_quarterly_income = pd.concat([pd.DataFrame([new_row]), d_quarterly_income], ignore_index=True)

d_time_series = d_time_series.iloc[::-1].reset_index(drop=False)
d_time_series = d_time_series.rename(columns={'index': 'date'})

last_date_time_series = d_time_series.iloc[len(d_time_series)-1, 0]
last_date_quarterly_income = d_quarterly_income.iloc[len(d_quarterly_income)-1, 0]
if last_date_time_series != last_date_quarterly_income:
    cut_off_date = max(last_date_quarterly_income, last_date_time_series)
    d_time_series = d_time_series[d_time_series['date'] >= cut_off_date]
    d_quarterly_income = d_quarterly_income[d_quarterly_income['date'] >= cut_off_date]


#-----------------------------------------------------------------------------------------------------------------------
# set unix time stamps
d_quarterly_income['date'] = pd.to_datetime(d_quarterly_income['date'])
d_quarterly_income['date'] = (d_quarterly_income['date'] - pd.Timestamp('1970-01-01')).dt.total_seconds()
d_time_series['date'] = pd.to_datetime(d_time_series['date'])
d_time_series['date'] = (d_time_series['date'] - pd.Timestamp('1970-01-01')).dt.total_seconds()

assert len(d_time_series) == len(d_quarterly_income)

# check data
for idx in range(len(d_time_series)):
    time_series_date = d_time_series.iloc[idx]['date']
    quarterly_income_date = d_quarterly_income.iloc[idx]['date']
    assert time_series_date == quarterly_income_date, f"DIFFERENCE FOUND d_time_series['date'] = {time_series_date}, d_quarterly_income['date'] = {quarterly_income_date} (Index: {idx})"

d_quarterly_income.to_excel(f'{directory_processed}d_quarterly_income.xlsx', index=True)
d_time_series.to_excel(f'{directory_processed}d_timeseries.xlsx', index=True)
