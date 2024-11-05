import numpy as np
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
import pandas as pd
from dotenv import load_dotenv
import os

# api key
load_dotenv()
API_KEY = os.getenv("API_KEY")

def api_raw_data_to_excel():
    # create instances
    api_d_fundamental_data = FundamentalData(key=API_KEY)  # load data_xlsx once then comment out
    api_d_time_series = TimeSeries(key=API_KEY, output_format='pandas')

    d_time_series, _ = api_d_time_series.get_daily(symbol='AAPL', outputsize='full')
    d_quarterly_income, _ = api_d_fundamental_data.get_income_statement_quarterly(symbol='AAPL')

    d_quarterly_income.to_excel('../../data_xlsx/d_quarterly_income_raw.xlsx', index=True)
    d_time_series.to_excel('../../data_xlsx/d_timeseries_raw.xlsx', index=True)

#api_raw_data_to_excel()

d_quarterly_income = pd.read_excel('../../data_xlsx/d_quarterly_income_raw.xlsx', index_col=0)
d_time_series = pd.read_excel('../../data_xlsx/d_timeseries_raw.xlsx', index_col=0) # 6288 days currently in there
r_time_series = d_time_series
d_time_series = d_time_series.reset_index()


# clean data_xlsx
d_quarterly_income = d_quarterly_income.drop(columns='depreciation')
d_quarterly_income = d_quarterly_income.drop(columns='reportedCurrency')
d_quarterly_income.replace("None", np.nan)
d_quarterly_income = d_quarterly_income.fillna(0) # replaces ever None with 0
d_quarterly_income['fiscalDateEnding'] = pd.to_datetime(d_quarterly_income['fiscalDateEnding'], errors='coerce')

def stretch_data(data_frame, column_name, months):
    new_df = pd.DataFrame(columns=data_frame.columns) # neue data_xlsx frame
    for i in range(months-1):
        row1 = data_frame.iloc[i]
        row2 = data_frame.iloc[i+1]

        upper_date = row1.iloc[0]
        lower_date = row2.iloc[0]

        upper_part = data_frame.iloc[:i]  # Teil bis idx2 (ohne idx2)
        #lower_part = data_frame.iloc[i:]  # Teil ab idx2

        current_date = lower_date #+ pd.Timedelta(days=1)
        row_to_use = row2.copy()

        while current_date < upper_date:
            row_to_use[column_name]=current_date
            upper_part.loc[len(upper_part)] = row_to_use # häng an den oberen teil unten neues dran
            current_date += pd.Timedelta(days=1)

        upper_part = upper_part.iloc[::-1].reset_index(drop=True)
        for i in range(0, len(upper_part)): # hänge alles vom oberen teil an new_df
            new_df.loc[len(new_df)] = upper_part.iloc[i]

    return new_df

counter_months = d_quarterly_income.iloc[:, 0].dt.to_period('M').nunique()
d_quarterly_income=stretch_data(d_quarterly_income, column_name = 'fiscalDateEnding', months=counter_months)
d_quarterly_income = d_quarterly_income.drop_duplicates(subset=['fiscalDateEnding']).sort_values(by='fiscalDateEnding').reset_index(drop=True)
d_quarterly_income = d_quarterly_income.iloc[::-1].reset_index(drop=True)

# set unix time stamps
d_quarterly_income['fiscalDateEnding'] = pd.to_datetime(d_quarterly_income['fiscalDateEnding'])
d_quarterly_income['fiscalDateEnding'] = (d_quarterly_income['fiscalDateEnding'] - pd.Timestamp('1970-01-01')).dt.total_seconds()
d_time_series['date'] = pd.to_datetime(d_time_series['date'])
d_time_series['date'] = (d_time_series['date'] - pd.Timestamp('1970-01-01')).dt.total_seconds()

d_quarterly_income.to_excel('../../data_xlsx/d_quarterly_income.xlsx', index=True)
d_time_series.to_excel('../../data_xlsx/d_timeseries.xlsx', index=True)
