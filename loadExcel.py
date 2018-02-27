from openpyxl import Workbook
from openpyxl import load_workbook
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time #clock, time methods


def getValidTickerIndexes(worksheet):
    columns = []
    row = worksheet['{}{}:{}{}'.format('A', 3, 'IYS', 3)]
    for n, cell in enumerate(row[0]):
        if n % 2 == 1 and cell.value is not None:
            columns.append(n)
    return columns


# unlike the other functions, this one should use worksheet "stock list" rather than the second ws
def findTickerIndex(ticker, worksheet):
    for n, row in enumerate(worksheet.iter_rows()):
        for cell in row:
            if str(cell.value) == ticker:
                return n * 2 + 1
    print(ticker + " not found.")
    return


# Excel indexes start with 1 rather than 0
def convertIndexToLetter(index):
    i = index
    letterIndex = ""

    while i > 0:
        remainder = (i - 1) % 26
        letterIndex = (chr((remainder) + ord('A'))) + letterIndex
        i = int((i - remainder)/26)

    return letterIndex


def getAllTimeSeries(worksheet, factor):
    indexes = getValidTickerIndexes(worksheet)
    for i in indexes:
        name = str(worksheet[convertIndexToLetter(i)+"1"].value)
        dates = []
        values = []
        row_range = 800
        letterIndex = convertIndexToLetter(i)
        for cell in worksheet['{}{}:{}{}'.format(letterIndex, 4, letterIndex, row_range)]:
            if cell[0].value is not None:
                dates.append(cell[0].value)
            else:
                break
        letterIndex = convertIndexToLetter(i+1)
        for cell in worksheet['{}{}:{}{}'.format(letterIndex, 4, letterIndex, row_range)]:
            if cell[0].value is not None:
                values.append(cell[0].value)
            else:
                break
        dates = np.array(dates, dtype = np.datetime64)
        try:
            data = pd.Series(values, index = dates)
            outputTimeSeries(data, factor, name)
        except ValueError:
            print(factor)
            print("Value Error for: " + name)
            print("# dates: " + str(len(dates)))
            print("# vals: " + str(len(values)))
    return


def outputTimeSeries(series, factor, name):
    dfSeries = pd.DataFrame({factor: series})
    try:
        df = pd.read_csv(name+".csv", index_col = 0)
        df.index = pd.to_datetime(df.index)
    except FileNotFoundError:
        df = pd.DataFrame()
    df = pd.concat([df, dfSeries], axis = 1)
    df.to_csv(name+".csv")
    return


def getTimeSeries(stock, worksheet, factor):
    stockList = wb["stock list"]
    i = findTickerIndex(stock, stockList)
    dates = []
    values = []
    row_range = 800
    letterIndex = convertIndexToLetter(i)
    for cell in worksheet['{}{}:{}{}'.format(letterIndex, 4, letterIndex, row_range)]:
        if cell[0].value is not None:
            dates.append(cell[0].value)
        else:
            break
    letterIndex = convertIndexToLetter(i+1)
    for cell in worksheet['{}{}:{}{}'.format(letterIndex, 4, letterIndex, row_range)]:
        if cell[0].value is not None:
            values.append(cell[0].value)
        else:
            break
    dates = np.array(dates, dtype = np.datetime64)
    data = pd.Series(values, index = dates, name = factor)
    return data


if __name__ == "__main__":
    factors = ["EPS Growth", "Trailing EPS", "Total Debt to Total Equity", "Volatility 180 Day", "Return on Invested Capital", "Return on Common Equity", "Return on Assets", "Volatility 360 Day", "Earnings Momentum", "Price to Cash Flow", "Price to Book Ratio", "EPS", "Dividend Yield", "Forward PE", "Volume", "Last Price"]
    filenames = list(map(lambda s: "USEquity(" + s + ").xlsm", factors))

    t0 = time.time()
    for f in filenames:
        wb = load_workbook(filename = f)
        sheet = wb.sheetnames[1]
        ws = wb[sheet]
        getAllTimeSeries(ws, sheet)
        print(time.time() - t0, "seconds wait time")
        wb.close()
