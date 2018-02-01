from openpyxl import Workbook
from openpyxl import load_workbook
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def getValidTickerIndexes(worksheet):
    columns = []
    for n, row in enumerate(worksheet.iter_cols(min_row = 3, max_col = len(list(worksheet.columns)), max_row = 3)):
        for cell in row:
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
            letterIndex += (chr((remainder-1) + ord('A')))
            i = i/26

    return ''.join(reversed(letterIndex))


def getAllTimeSeries(worksheet):
    indexes = getValidTickerIndexes(worksheet)
    for i in indexes:
        dates = []
        values = []
        row_range = 100
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
        data = pd.Series(values, index = dates)
        print i


def getTimeSeries(stock, worksheet, factor):
    stockList = wb["stock list"]
    i = findTickerIndex(stock, stockList)
    dates = []
    values = []
    row_range = 400
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
    data = pd.Series(values, index = dates)
    return data


if __name__ == "__main__":
    wb = load_workbook(filename = "USEquity(Forward PE).xlsm")
    ws = wb["Forward PE"]
    getAllTimeSeries(ws)
    plt.show()
