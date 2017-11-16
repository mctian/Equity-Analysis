import requests
import numpy as np
import pandas as pd
import quandl
import datetime
import csv
from datapackage import Package

package = Package('http://datahub.io/core/s-and-p-500-companies/datapackage.json')

# get list of resources:
resources = package.descriptor['resources']
resourceList = [resources[x]['name'] for x in range(0, len(resources))]
print(resourceList)

data = package.resources[0].read()
print(data)

def get_price_data(t):
    data = quandl.get('WIKI/'+t)
    return data


def main():
    universe = get_all_tickers()
    indexes = [2]
    for i in indexes:
        print(get_price_data(universe[i]))
    return


def get_all_tickers():
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
    print(yesterday)
    data = quandl.get('WIKI', paginate=True)
    print(data)
    return data

if __name__ == "__main__":
    quandl.ApiConfig.api_key = "RxidFKB69HRV8VFHbXqM"
    main()

    #r = requests.get('http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=NASDAQ&render=download')