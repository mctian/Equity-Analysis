import requests
import numpy as np
import pandas as pd
import quandl
import datetime
import csv
from datapackage import Package

def get_price_data(t):
    data = quandl.get('WIKI/'+t)
    return data


def main():
    universe = get_all_tickers()
    indexes = [2]
    for i in indexes:
        print(get_price_data(universe[i]))
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

def get_all_tickers():
    package = Package('http://datahub.io/core/s-and-p-500-companies/datapackage.json')
    resources = package.descriptor['resources']
    resourceList = [resources[x]['name'] for x in range(0, len(resources))]
    print(resourceList)
    data = package.resources[0].read()
    print(data)
    tickers = []
    return tickers


if __name__ == "__main__":
    quandl.ApiConfig.api_key = "RxidFKB69HRV8VFHbXqM"
    main()

    #r = requests.get('http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=NASDAQ&render=download')