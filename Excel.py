from openpyxl import Workbook
from openpyxl import load_workbook
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def getValidTickerIndexes(worksheet):
    columns = []
    for n, col in enumerate(worksheet.iter_cols(min_row = 3, max_col = len(list(worksheet.columns)), max_row = 3)):
        for cell in col:
            if (n + 1) % 2 == 1 and cell.value is not None:
                columns.append(n + 1)
    return columns


# unlike the other functions, this one should use worksheet "stock list" rather than the second ws
def findTickerIndex(ticker, worksheet):
    for n, row in enumerate(worksheet.iter_rows()):
        for cell in row:
            if str(cell.value) == ticker:
                return n * 2 + 1
    print(ticker + " not found.")

# Excel indexes start with 1 rather than 0
def convertIndexToLetter(index):
    i = index
    letterIndex = ""

    while i > 0:
        remainder = i % 26
        if remainder == 0:
            letterIndex += 'Z'
            i = i/26 - 1
        else:
            letterIndex += chr((remainder-1) + ord('A'))
            i = i/26

    return ''.join(reversed(letterIndex))


def getAllTimeSeries(worksheet):
    indexes = getValidTickerIndexes(worksheet)
    for i in indexes:
        dates = []
        values = []
        row_range = len(worksheet[i])
        endReached = False
        letterIndex = convertIndexToLetter(i)
        for row in worksheet.iter_rows('{}{}:{}{}'.format(letterIndex, 4, letterIndex, row_range)):
            for cell in row:
                if cell.value is not None:
                    dates.append(cell.value)
                else:
                    endReached = True
                    break
            if endReached is True:
                endReached = False
                break
        letterIndex = convertIndexToLetter(i+1)
        for row in worksheet.iter_rows('{}{}:{}{}'.format(letterIndex, 4, letterIndex, row_range)):
            for cell in row:
                if cell.value is not None:
                    values.append(cell.value)
                else:
                    endReached = True
                    break
            if endReached is True:
                endReached = False
                break
        date = np.array(dates, dtype = np.datetime64)
        data = pd.Series(values, index = dates)
        #at this point, we would store the data somewhere, but we don't have a database :(


def getTimeSeries(stock, worksheet, factor):
    stockList = wb["stock list"]
    i = findTickerIndex(stock, stockList)
    dates = []
    values = []
    row_range = len(worksheet[i])
    endReached = False
    letterIndex = convertIndexToLetter(i)
    for row in worksheet.iter_rows('{}{}:{}{}'.format(letterIndex, 4, letterIndex, row_range)):
        for cell in row:
            if cell.value is not None:
                dates.append(cell.value)
            else:
                endReached = True
                break
        if endReached is True:
            endReached = False
            break
    letterIndex = convertIndexToLetter(i+1)
    for row in worksheet.iter_rows('{}{}:{}{}'.format(letterIndex, 4, letterIndex, row_range)):
        for cell in row:
            if cell.value is not None:
                values.append(cell.value)
            else:
                endReached = True
                break
        if endReached is True:
            endReached = False
            break
    date = np.array(dates, dtype = np.datetime64)
    data = pd.Series(values, name = factor, index = dates)
    return data


if __name__ == "__main__":
    wb = load_workbook(filename = "USEquity(Dividend Yield).xlsm")
    ws = wb["Dividend Yield"]
    dividends = getTimeSeries("AAPL", ws, "Dividend Yield")
    wb = load_workbook(filename = "USEquity(Forward PE).xlsm")
    ws = wb["Forward PE"]
    forward_pe = getTimeSeries("AAPL", ws, "Forward PE")
    dividends.plot(legend=True)
    forward_pe.plot(legend=True)
    plt.show()
