import requests
import numpy as np
import pandas as pd
import quandl
import datetime
import csv
from datapackage import Package


def get_data(t):
    data = quandl.get('WIKI/'+t)
    return data


def get_price_data(t, start, end):
    data = quandl.get('Wiki/'+ t + '.4', start_date=start, end_date=end)
    return data


def main():
    universe = get_all_tickers()
    indexes = [2]
    for i in indexes:
        print(get_data(universe[i]))
    return


def get_today():
    today = datetime.datetime.now()
    date_today = str(today.year)
    if today.month < 10:
        date_today += '-0' + str(today.month) + '-'
    else:
        date_today += '-' + str(today.month) + '-'
    if today.day < 10:
        date_today += '0' + str(today.day)
    else:
        date_today += str(today.day)
    return date_today


def get_yesterday():
    today = datetime.datetime.now()
    yesterday = str(today.year)
    if today.month < 10:
        yesterday += '-0' + str(today.month) + '-'
    else:
        yesterday += '-' + str(today.month) + '-'
    if today.day < 10:
        yesterday += '0' + str(today.day)
    else:
        yesterday += str(today.day - 1)
    return yesterday


def format_date(month, day, year):
    date = str(year)
    if month < 10:
        date += '-0' + str(month) + '-'
    else:
        date += '-' + str(month) + '-'
    if day < 10:
        date += '0' + str(day)
    else:
        date += str(day)
    return date


def get_all_tickers():
    package = Package('http://datahub.io/core/s-and-p-500-companies/datapackage.json')
    data = package.resources[0].read()
    tickers = []
    for i in range(0,len(data)):
        tickers.append(data[i][0])
    return tickers


if __name__ == "__main__":
    quandl.ApiConfig.api_key = "RxidFKB69HRV8VFHbXqM"
    main()
