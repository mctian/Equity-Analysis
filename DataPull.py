import requests
import numpy as np
import pandas as pd
import quandl
import datetime
from datapackage import Package
import holidays


quandl.ApiConfig.api_key = "RxidFKB69HRV8VFHbXqM"


def get_data(t):
    data = quandl.get('WIKI/'+t)
    return data


def get_price_data(t, start, end):
    start_date = format_date(start)
    end_date = format_date(end)
    data = quandl.get('Wiki/'+ t, start_date=start_date, end_date=end_date)
    return data


def format_date(date):
    formatted = str(date.year)
    if date.month < 10:
        formatted += '-0' + str(date.month) + '-'
    else:
        formatted += '-' + str(date.month) + '-'
    if date.day < 10:
        formatted += '0' + str(date.day)
    else:
        formatted += str(date.day)
    return formatted


def market_days(start, end):
    all_holidays = list((holidays.US(state='NY', years = datetime.datetime.today().year)+holidays.EuropeanCentralBank()).keys())
    indexes = [1, 4, 6, 7, 11, 12, 13, 14, 18]
    nyse_holidays = []
    for i in indexes:
        nyse_holidays.append(all_holidays[i])
    days = np.busday_count(start, end, holidays=nyse_holidays)
    return days


def check_open(date):
    all_holidays = list((holidays.US(state='NY', years=datetime.datetime.today().year) + holidays.EuropeanCentralBank()).keys())
    indexes = [1, 4, 6, 7, 11, 12, 13, 14, 18]
    for i in indexes:
        if date == all_holidays[i]:
            return False
    if date.weekday() > 4:
        return False
    return True


def get_all_tickers():
    package = Package('http://datahub.io/core/s-and-p-500-companies/datapackage.json')
    data = package.resources[0].read()
    tickers = []
    for i in range(0,len(data)):
        tickers.append(data[i][0])
    return tickers
