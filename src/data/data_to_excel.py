import numpy as np
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
import pandas as pd
from dotenv import load_dotenv
import os
import requests
import yfinance as yf

# API KEY
load_dotenv()
API_KEY = os.getenv("API_KEY")

directory_raw = r'C:\\Users\\LeonardSchickedanz\\PycharmProjects\\PredictStockPrice\data\raw'
directory_processed = r'C:\\Users\\LeonardSchickedanz\\PycharmProjects\\PredictStockPrice\\data\\processed'

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

economic_indicators = (
        'd_real_gdp',
        'd_real_gdp_per_capita',
        'd_federal_funds_rate',
        'd_cpi',
        'd_inflation',
        'd_retail_sales',
        'd_durables',
        'd_unemployment',
        'd_nonfarm_payroll'
        )

def economic_indicators_to_excel():
    main_url = 'https://www.alphavantage.co/query?function='
    api_key_url = f'&apikey={API_KEY}'

    url_list = (
        'REAL_GDP&interval=annual',
        'REAL_GDP_PER_CAPITA',
        'FEDERAL_FUNDS_RATE&interval=monthly',
        'CPI&interval=monthly',
        'INFLATION',
        'RETAIL_SALES',
        'DURABLES',
        'UNEMPLOYMENT',
        'NONFARM_PAYROLL'
    )

    dataframes = []

    for idx1, u in enumerate(url_list):
        try:
            url =  f'{main_url}{u}{api_key_url}'
            r = requests.get(url)
            data = r.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {u}: {e}")
            return

        if 'Information' in data and 'rate limit' in data['Information'].lower():
            print("OUT OF API REQUESTS")
            return
        else:
            print("API request successfull")

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

        df.to_excel(fr'{directory_raw}\\{economic_indicators[idx1]}_raw.xlsx', index=True)

DATA_LIST = [
    pd.read_excel(f'{directory_raw}/d_real_gdp_raw.xlsx', index_col=0),
    pd.read_excel(f'{directory_raw}/d_real_gdp_per_capita_raw.xlsx', index_col=0),
    pd.read_excel(f'{directory_raw}/d_federal_funds_rate_raw.xlsx', index_col=0),
    pd.read_excel(f'{directory_raw}/d_cpi_raw.xlsx', index_col=0),
    pd.read_excel(f'{directory_raw}/d_inflation_raw.xlsx', index_col=0),
    pd.read_excel(f'{directory_raw}/d_retail_sales_raw.xlsx', index_col=0),
    pd.read_excel(f'{directory_raw}/d_durables_raw.xlsx', index_col=0),
    pd.read_excel(f'{directory_raw}/d_unemployment_raw.xlsx', index_col=0),
    pd.read_excel(f'{directory_raw}/d_nonfarm_payroll_raw.xlsx', index_col=0),
    pd.read_excel(f'{directory_raw}/d_quarterly_income_raw.xlsx', index_col=0),
    pd.read_excel(f'{directory_raw}/d_timeseries_raw.xlsx', index_col=0),
    ]

def stretch_data(data, min_date, max_date, date_column = 'date'):
    data.set_index(date_column, inplace=True)
    #data.sort_index(inplace=True)

    data = data.resample('D').ffill()

    date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    data = data.reindex(date_range)
    data = data.reset_index(names=['date'])

    data = data.iloc[::-1].reset_index(drop=True)

    return data

def date_to_unix_time_stamp(data, date_column='date'):
    data[date_column] = pd.to_datetime(data[date_column])
    data[date_column] = (data[date_column] - pd.Timestamp('1970-01-01')).dt.total_seconds()
    return data

def main(ticker):

    economic_indicators_to_excel()
    api_raw_data_to_excel(ticker)

    # CLEAN DATA
    d_quarterly_income = DATA_LIST[9]
    d_time_series = DATA_LIST[10]

    d_time_series = d_time_series.iloc[2:].reset_index(drop=False)
    d_time_series.columns = ['date', 'close', 'high', 'low', 'open', 'volume']
    d_time_series = d_time_series[::-1].reset_index(drop=True)

    d_quarterly_income = d_quarterly_income.drop(columns='depreciation')
    d_quarterly_income = d_quarterly_income.drop(columns='reportedCurrency')
    d_quarterly_income.replace("None", np.nan)
    d_quarterly_income = d_quarterly_income.fillna(0) # replaces every None with 0
    d_quarterly_income.rename(columns={'fiscalDateEnding': 'date'}, inplace=True)


    DATA_LIST[9] = d_quarterly_income
    DATA_LIST[10] = d_time_series

    for idx in range(len(DATA_LIST)-2):
        DATA_LIST[idx]['value'] = DATA_LIST[idx].rename(columns={f'value': f'{economic_indicators[idx].lstrip('d_')}'}, inplace=True)


    for df in DATA_LIST:
        df['date'] = pd.to_datetime(df['date'])
    date_columns = [df['date'] for df in DATA_LIST]
    max_d = min(df['date'].max() for df in DATA_LIST) # currently 2023-01-01
    min_d = max(df['date'].min() for df in DATA_LIST)

    # UNIX TIME STAMPS AND STRETCH DATA
    for idx1, df1 in enumerate(DATA_LIST):
        DATA_LIST[idx1] = stretch_data(data=df1,min_date=min_d ,max_date=max_d)
        DATA_LIST[idx1] = date_to_unix_time_stamp(DATA_LIST[idx1])

    DATA_LIST[10].to_excel(r'C:\Users\LeonardSchickedanz\PycharmProjects\PredictStockPrice\data\d_timeseries_raw3.xlsx')

    # CHECK DATA
    assertion_length = len(DATA_LIST[0]['date'])
    for idx2, df2 in enumerate(DATA_LIST):
        if len(df2) != assertion_length:
            raise Exception(f'DATA DOES NOT HAVE SAME LENGTH {idx2}')

    for idx3 in range(assertion_length):
       assert (
           DATA_LIST[0]['date'][idx3] == DATA_LIST[1]['date'][idx3] and
           DATA_LIST[0]['date'][idx3] == DATA_LIST[2]['date'][idx3] and
           DATA_LIST[0]['date'][idx3] == DATA_LIST[3]['date'][idx3] and
           DATA_LIST[0]['date'][idx3] == DATA_LIST[4]['date'][idx3] and
           DATA_LIST[0]['date'][idx3] == DATA_LIST[5]['date'][idx3] and
           DATA_LIST[0]['date'][idx3] == DATA_LIST[6]['date'][idx3] and
           DATA_LIST[0]['date'][idx3] == DATA_LIST[7]['date'][idx3] and
           DATA_LIST[0]['date'][idx3] == DATA_LIST[8]['date'][idx3] and
           DATA_LIST[0]['date'][idx3] == DATA_LIST[9]['date'][idx3] and
           DATA_LIST[0]['date'][idx3] == DATA_LIST[10]['date'][idx3]
       ), f'DIFFERENCE FOUND AT INDEX={idx3}'

    # DATA TO EXCEL
    for idx in range(len(DATA_LIST)-2):
        DATA_LIST[idx].to_excel(f'{directory_processed}/{economic_indicators[idx]}.xlsx')
    DATA_LIST[9].to_excel(f'{directory_processed}/d_quarterly_income.xlsx')
    DATA_LIST[10].to_excel(f'{directory_processed}/d_timeseries.xlsx')
